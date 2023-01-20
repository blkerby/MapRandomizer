mod map_tiles;

use std::path::Path;

use crate::{
    game_data::{GameData, Item, Map, NodePtr, DoorPtr},
    randomize::Randomization,
};
use anyhow::{ensure, Context, Result};
use hashbrown::HashMap;
use ips;
use std::iter;

const NUM_AREAS: usize = 6;

type AsmPtr = usize;

fn snes2pc(addr: usize) -> usize {
    addr >> 1 & 0x3F8000 | addr & 0x7FFF
}

#[derive(Clone)]
pub struct Rom {
    pub data: Vec<u8>,
}

impl Rom {
    pub fn load(path: &Path) -> Result<Self> {
        let data = std::fs::read(path)
            .with_context(|| format!("Unable to load ROM at path {}", path.display()))?;
        Ok(Rom { data })
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
        Ok(())
    }

    pub fn write_u16(&mut self, addr: usize, x: isize) -> Result<()> {
        ensure!(
            addr + 2 <= self.data.len(),
            "write_u16 address out of bounds"
        );
        ensure!(x >= 0 && x <= 0xFFFF, "write_u16 data does not fit");
        self.data[addr] = (x & 0xFF) as u8;
        self.data[addr + 1] = (x >> 8) as u8;
        Ok(())
    }

    pub fn write_u24(&mut self, addr: usize, x: isize) -> Result<()> {
        ensure!(
            addr + 3 <= self.data.len(),
            "write_u24 address out of bounds"
        );
        ensure!(x >= 0 && x <= 0xFFFFFF, "write_u24 data does not fit");
        self.data[addr] = (x & 0xFF) as u8;
        self.data[addr + 1] = ((x >> 8) & 0xFF) as u8;
        self.data[addr + 2] = (x >> 16) as u8;
        Ok(())
    }

    pub fn write_n(&mut self, addr: usize, x: &[u8]) -> Result<()> {
        ensure!(
            addr + x.len() <= self.data.len(),
            "write_n address out of bounds"
        );
        self.data[addr..(addr + x.len())].copy_from_slice(x);
        Ok(())
    }
}

pub struct Patcher<'a> {
    pub orig_rom: &'a mut Rom,
    pub rom: &'a mut Rom,
    pub randomization: &'a Randomization,
    pub game_data: &'a GameData,
    pub map: &'a Map,
}

fn xy_to_map_offset(x: isize, y: isize) -> isize {
    let y1 = y + 1;
    if x < 32 {
        (y1 * 32 + x) * 2
    } else {
        ((y1 + 32) * 32 + x - 32) * 2
    }
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

fn apply_orig_ips_patches(rom: &mut Rom) -> Result<()> {
    let patches_dir = Path::new("../patches/ips/");
    let patches = vec![
        "mb_barrier",
        "mb_barrier_clear",
        "hud_expansion_opaque",
        "gray_doors",
    ];
    for patch_name in patches {
        let patch_path = patches_dir.join(patch_name.to_string() + ".ips");
        apply_ips_patch(rom, &patch_path)?;
    }
    Ok(())
}

impl<'a> Patcher<'a> {
    fn apply_ips_patches(&mut self) -> Result<()> {
        let patches_dir = Path::new("../patches/ips/");
        let mut patches = vec![
            "vanilla_bugfixes",
            "music",
            "crateria_sky_fixed",
            "everest_tube",
            "sandfalls",
            "saveload",
            "map_area",
            "elevators_speed",
            "boss_exit",
            "itemsounds",
            "progressive_suits",
            "disable_map_icons",
            "escape",
            "mother_brain_no_drain",
            "tourian_map",
            "tourian_eye_door",
            "no_explosions_before_escape",
            "escape_room_1",
            "unexplore",
            "max_ammo_display",
            "missile_refill_all",
            "sound_effect_disables",
            "title_map_animation",
            "fast_reload",
        ];
        patches.push("new_game_extra");
        // "new_game_extra' if args.debug else 'new_game",
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
        extra_door_asm_map: &HashMap<NodePtr, (AsmPtr, AsmPtr)>,
    ) -> Result<()> {
        let door_data = self.orig_rom.read_n(dst_entrance_ptr, 12)?;
        self.rom.write_n(src_exit_ptr, door_data)?;
        if let Some(&(new_asm, end_asm)) = extra_door_asm_map.get(&src_exit_ptr) { 
            // Set extra custom ASM applicable to exiting from the given door exit:
            self.rom.write_u16(src_exit_ptr + 10, new_asm as isize)?;

            // Patch the custom ASM to jump to original ASM for the given door entrance:
            let orig_asm = self.orig_rom.read_u16(dst_entrance_ptr + 10)?;
            if orig_asm == 0 {
                // There is no original ASM for this entrance, so just return:
                self.rom.write_u8(snes2pc(0x8F8000 | end_asm), 0x60)?;  // RTS
            } else {
                let jmp_asm = vec![0x4C, (orig_asm & 0xFF) as u8, (orig_asm >> 8) as u8]; // JMP orig_asm
                self.rom.write_n(snes2pc(0x8F8000 | end_asm), &jmp_asm)?;
            }
        }
        Ok(())
    }

    // Returns map from door data PC address to 1) new custom door ASM pointer, 2) end of custom door ASM
    // where an RTS or JMP instruction must be added (based on the connecting door).
    fn prepare_extra_door_asm(&mut self) -> Result<HashMap<DoorPtr, (AsmPtr, AsmPtr)>> {
        let toilet_exit_asm: Vec<u8> = vec![0x20, 0x01, 0xE3];  // JSR 0xE301
        let boss_exit_asm: Vec<u8> = vec![0x20, 0xF0, 0xF7];  // JSR 0xF7F0
        let extra_door_asm: Vec<(DoorPtr, Vec<u8>)> = vec![
            (0x1A600, toilet_exit_asm.clone()),  // Aqueduct toilet door down
            (0x1A60C, toilet_exit_asm.clone()),  // Aqueduct toilet door up
            (0x191CE, boss_exit_asm.clone()),  // Kraid left exit
            (0x191DA, boss_exit_asm.clone()),  // Kraid right exit
            (0x1A96C, boss_exit_asm.clone()),  // Draygon left exit
            (0x1A978, boss_exit_asm.clone()),  // Draygon right exit
            (0x193DE, boss_exit_asm.clone()),  // Crocomire left exit
            (0x193EA, boss_exit_asm.clone()),  // Crocomire top exit
            (0x1A2C4, boss_exit_asm.clone()),  // Phantoon exit
        ];

        let mut door_asm_free_space = 0xEE10;  // in bank 0x8F
        let mut extra_door_asm_map: HashMap<DoorPtr, (AsmPtr, AsmPtr)> = HashMap::new();
        for &(door_ptr, ref asm) in &extra_door_asm {
            extra_door_asm_map.insert(door_ptr, (door_asm_free_space, door_asm_free_space + asm.len()));
            self.rom.write_n(snes2pc(0x8F8000 | door_asm_free_space), asm)?;
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
            if src_exit_ptr.is_some() && dst_entrance_ptr.is_some() {
                self.write_one_door_data(src_exit_ptr.unwrap(), dst_entrance_ptr.unwrap(), &extra_door_asm_map)?;
            }
            if dst_exit_ptr.is_some() && src_entrance_ptr.is_some() {
                self.write_one_door_data(dst_exit_ptr.unwrap(), src_entrance_ptr.unwrap(), &extra_door_asm_map)?;
            }
        }
        Ok(())
    }

    fn fix_save_stations(&mut self) -> Result<()> {
        let save_station_ptrs = vec![
            0x44C5, 0x44D3, 0x45CF, 0x45DD, 0x45EB, 0x45F9, 0x4607, 0x46D9, 0x46E7, 0x46F5, 0x4703,
            0x4711, 0x471F, 0x481B, 0x4917, 0x4925, 0x4933, 0x4941, 0x4A2F, 0x4A3D,
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
            let orig_entrance_door_ptr = (self.orig_rom.read_u16(ptr + 2)? + 0x10000) as NodePtr;
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
        let area_map_ptrs: Vec<isize> = vec![
            0x1A9000, // Crateria
            0x1A8000, // Brinstar
            0x1AA000, // Norfair
            0x1AB000, // Wrecked ship
            0x1AC000, // Maridia
            0x1AD000, // Tourian
        ];

        // Determine upper-left corner of each area:
        let mut area_map_min_x = [isize::MAX; NUM_AREAS];
        let mut area_map_max_x = [0; NUM_AREAS];
        let mut area_map_min_y = [isize::MAX; NUM_AREAS];
        let mut area_map_max_y = [0; NUM_AREAS];
        for i in 0..self.map.area.len() {
            let area = self.map.area[i];
            let x = self.map.rooms[i].0 as isize;
            let y = self.map.rooms[i].1 as isize;
            if x < area_map_min_x[area] {
                area_map_min_x[area] = x;
            }
            if x > area_map_max_x[area] {
                area_map_max_x[area] = x;
            }
            if y < area_map_min_y[area] {
                area_map_min_y[area] = y;
            }
            if y > area_map_max_y[area] {
                area_map_max_y[area] = y;
            }
        }

        // Clear all map tilemap data:
        for area_ptr in &area_map_ptrs {
            for i in 0..(64 * 32) {
                self.rom.write_u16((area_ptr + i * 2) as usize, 0x001F)?;
            }
        }

        // Write new map tilemap data (and room X & Y map position) by room:
        for (i, room) in self.game_data.room_geometry.iter().enumerate() {
            let orig_area = self.orig_rom.read_u8(room.rom_address + 1)? as usize;
            let orig_base_x = self.orig_rom.read_u8(room.rom_address + 2)?;
            let mut orig_base_y = self.orig_rom.read_u8(room.rom_address + 3)?;
            if room.name == "Aqueduct" {
                orig_base_y -= 4;
            }
            let orig_base_ptr = area_map_ptrs[orig_area];
            let new_area = self.map.area[i];
            let new_base_ptr = area_map_ptrs[new_area];
            let new_margin_x = (64 - (area_map_max_x[new_area] - area_map_min_x[new_area])) / 2;
            let new_margin_y = (32 - (area_map_max_y[new_area] - area_map_min_y[new_area])) / 2;
            let new_base_x = self.map.rooms[i].0 as isize - area_map_min_x[new_area] + new_margin_x;
            let new_base_y = self.map.rooms[i].1 as isize - area_map_min_y[new_area] + new_margin_y;
            assert!(new_base_x >= 2);
            assert!(new_base_y >= 1);
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
            let ptr = self.orig_rom.read_u16(room_ptr + pos)? as usize;
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

    fn remove_non_blue_doors(&mut self) -> Result<()> {
        let plm_types_to_remove = vec![
            0xC88A, 0xC85A, 0xC872, // right pink/yellow/green door
            0xC890, 0xC860, 0xC878, // left pink/yellow/green door
            0xC896, 0xC866, 0xC87E, // down pink/yellow/green door
            0xC89C, 0xC86C, 0xC884, // up pink/yellow/green door
            0xDB48, 0xDB4C, 0xDB52, 0xDB56, 0xDB5A, 0xDB60, // eye doors
            0xC8CA, // wall in Escape Room 1
        ];
        let gray_door_plm_types = vec![
            0xC848, // left gray door
            0xC842, // right gray door
            0xC854, // up gray door
            0xC84E, // down gray door
        ];
        let keep_gray_door_room_names: Vec<String> = vec![
            "Kraid Room",
            "Draygon's Room",
            "Ridley's Room",
            "Golden Torizo's Room",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        for room in &self.game_data.room_geometry {
            let state_ptrs = self.get_room_state_ptrs(room.rom_address)?;
            for &(event_ptr, state_ptr) in &state_ptrs {
                let plm_set_ptr = self.rom.read_u16(state_ptr + 20)? as usize;
                let mut ptr = plm_set_ptr + 0x70000;
                loop {
                    let plm_type = self.orig_rom.read_u16(ptr)?;
                    if plm_type == 0 {
                        break;
                    }
                    let room_keep_gray_door = keep_gray_door_room_names.contains(&room.name)
                        || (room.name == "Pit Room" && event_ptr == 0xE652);
                    let is_removable_grey_door = gray_door_plm_types.contains(&plm_type) && !room_keep_gray_door;
                    if plm_types_to_remove.contains(&plm_type) || is_removable_grey_door {
                        self.rom.write_u16(ptr, 0xB63B)?;  // right continuation arrow (should have no effect, giving a blue door)
                        self.rom.write_u16(ptr + 2, 0)?;   // position = (0, 0)
                    }
                    ptr += 6;
                }
            }
        }
        Ok(())
    }
}

pub fn make_rom(
    base_rom_path: &Path,
    randomization: &Randomization,
    game_data: &GameData,
) -> Result<Rom> {
    let mut orig_rom = Rom::load(base_rom_path)?;
    apply_orig_ips_patches(&mut orig_rom)?;
    map_tiles::MapPatcher::new(&mut orig_rom, game_data).apply_patches()?;

    let mut rom = orig_rom.clone();
    let mut patcher = Patcher {
        orig_rom: &mut orig_rom,
        rom: &mut rom,
        randomization,
        map: &randomization.map,
        game_data,
    };
    patcher.apply_ips_patches()?;
    patcher.place_items()?;
    patcher.write_door_data()?;
    patcher.fix_save_stations()?;
    patcher.write_map_tilemaps()?;
    patcher.write_map_areas()?;
    patcher.make_map_revealed()?;
    patcher.remove_non_blue_doors()?;
    // TODO: add CRE reload for Kraid & Crocomire
    Ok(rom)
}
