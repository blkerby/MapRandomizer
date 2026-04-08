pub mod mosaic;
pub mod retiling;
pub mod room_palettes;
pub mod samus_sprite;
pub mod vanilla_music;

use anyhow::{Result, bail};
use log::info;
use std::cmp::min;
use std::path::Path;

use crate::patch::glowpatch_writer::write_glowpatch;
use crate::patch::{Rom, apply_ips_patch, snes2pc, write_credits_big_char};
use maprando_game::{GameData, Map};
use mosaic::MosaicTheme;
use retiling::apply_retiling;
use room_palettes::apply_area_themed_palettes;
use samus_sprite::SamusSpriteCategory;

struct AllocatorBlock {
    start_addr: usize,
    end_addr: usize,
    current_addr: usize,
}

pub struct Allocator {
    blocks: Vec<AllocatorBlock>,
}

impl Allocator {
    pub fn new(blocks: Vec<(usize, usize)>) -> Self {
        Allocator {
            blocks: blocks
                .into_iter()
                .map(|(start, end)| AllocatorBlock {
                    start_addr: start,
                    end_addr: end,
                    current_addr: start,
                })
                .collect(),
        }
    }

    pub fn allocate(&mut self, size: usize) -> Result<usize> {
        for block in &mut self.blocks {
            if block.end_addr - block.current_addr >= size {
                let addr = block.current_addr;
                block.current_addr += size;
                return Ok(addr);
            }
        }
        bail!("Failed to allocate {} bytes", size);
    }

    pub fn get_stats(&self) -> (usize, usize, usize) {
        let mut min_free = 0; // only count completely free blocks
        let mut max_free = 0; // include partially free blocks
        let mut total_capacity = 0;
        for block in &self.blocks {
            total_capacity += block.end_addr - block.start_addr;
            if block.current_addr == block.start_addr {
                min_free += block.end_addr - block.start_addr;
            }
            max_free += block.end_addr - block.current_addr;
        }
        (min_free, max_free, total_capacity)
    }
}

#[derive(Debug)]
pub enum MusicSettings {
    AreaThemed,
    Disabled,
}

#[derive(Debug)]
pub enum PaletteTheme {
    Vanilla,
    AreaThemed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TileTheme {
    Vanilla,
    AreaThemed,
    Scrambled,
    Constant(String),
}

#[derive(Default, Debug, Copy, Clone)]
pub enum DoorTheme {
    #[default]
    Vanilla,
    Alternate,
}

#[derive(Default, Debug, Copy, Clone)]
pub enum ControllerButton {
    #[default]
    Default,
    Left,
    Right,
    Up,
    Down,
    X,
    Y,
    A,
    B,
    L,
    R,
    Select,
    Start,
}

#[derive(Default, Debug)]
pub struct ControllerConfig {
    pub shot: ControllerButton,
    pub jump: ControllerButton,
    pub dash: ControllerButton,
    pub item_select: ControllerButton,
    pub item_cancel: ControllerButton,
    pub angle_up: ControllerButton,
    pub angle_down: ControllerButton,
    pub spin_lock_buttons: Vec<ControllerButton>,
    pub quick_reload_buttons: Vec<ControllerButton>,
    pub moonwalk: bool,
}

#[derive(Debug, Copy, Clone)]
pub enum ShakingSetting {
    Vanilla,
    Reduced,
    Disabled,
}

#[derive(Debug, Copy, Clone)]
pub enum FlashingSetting {
    Vanilla,
    Reduced,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ItemDotChange {
    Disabled,
    Fade,
    Disappear,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StatuesHallwayTiling {
    Disabled,
    Default,
    Enabled,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StatuesHallwayAudio {
    Disabled,
    Enabled,
    Louder,
}

#[derive(Debug)]
pub struct CustomizeSettings {
    pub samus_sprite: Option<String>,
    pub etank_color: Option<(u8, u8, u8)>,
    pub item_dot_change: ItemDotChange,
    pub transition_letters: bool,
    pub save_icons: bool,
    pub boss_icons: bool,
    pub miniboss_icons: bool,
    pub reserve_hud_style: bool,
    pub vanilla_screw_attack_animation: bool,
    pub palette_theme: PaletteTheme,
    pub tile_theme: TileTheme,
    pub door_theme: DoorTheme,
    pub music: MusicSettings,
    pub disable_beeping: bool,
    pub shaking: ShakingSetting,
    pub flashing: FlashingSetting,
    pub room_names: bool,
    pub statues_hallway_tiling: StatuesHallwayTiling,
    pub statues_hallway_audio: StatuesHallwayAudio,
    pub controller_config: ControllerConfig,
}

impl Default for CustomizeSettings {
    fn default() -> Self {
        Self {
            samus_sprite: Some("samus_vanilla".to_string()),
            etank_color: None,
            item_dot_change: ItemDotChange::Fade,
            transition_letters: true,
            save_icons: true,
            boss_icons: true,
            miniboss_icons: true,
            reserve_hud_style: true,
            vanilla_screw_attack_animation: true,
            room_names: true,
            palette_theme: PaletteTheme::Vanilla,
            tile_theme: TileTheme::AreaThemed,
            door_theme: DoorTheme::Vanilla,
            music: MusicSettings::AreaThemed,
            disable_beeping: false,
            shaking: ShakingSetting::Vanilla,
            flashing: FlashingSetting::Vanilla,
            statues_hallway_tiling: StatuesHallwayTiling::Default,
            statues_hallway_audio: StatuesHallwayAudio::Enabled,
            controller_config: ControllerConfig::default(),
        }
    }
}

fn remove_mother_brain_flashing(rom: &mut Rom) -> Result<()> {
    // Disable start of flashing after Mother Brain 1:
    rom.write_u16(snes2pc(0xA9CFFE), 0)?;

    // Disable end of flashing (to prevent palette from getting overwritten)
    rom.write_u8(snes2pc(0xA9D00C), 0x60)?; // RTS

    Ok(())
}

fn apply_custom_samus_sprite(
    rom: &mut Rom,
    settings: &CustomizeSettings,
    samus_sprite_categories: &[SamusSpriteCategory],
) -> Result<()> {
    if settings.samus_sprite.is_some() || !settings.vanilla_screw_attack_animation {
        let sprite_name = settings
            .samus_sprite
            .clone()
            .unwrap_or("samus_vanilla".to_string());
        let patch_path_str = format!("../patches/samus_sprites/{sprite_name}.ips");
        apply_ips_patch(rom, Path::new(&patch_path_str))?;

        if settings.vanilla_screw_attack_animation {
            // Disable spin attack animation, to make it behave like vanilla: Screw attack animation will look like
            // you have Space Jump even if you don't:
            rom.write_u16(snes2pc(0x9B93FE), 0)?;
        }
    }

    // Patch credits to give credit to the sprite author:
    if let Some(sprite_name) = settings.samus_sprite.as_ref() {
        for category in samus_sprite_categories {
            for info in &category.sprites {
                if &info.name == sprite_name {
                    // Write the sprite name
                    let mut chars = vec![];
                    let credits_name = info
                        .credits_name
                        .clone()
                        .unwrap_or(info.display_name.clone());
                    for c in credits_name.chars() {
                        let c = c.to_ascii_uppercase();
                        if c.is_ascii_uppercase() || c == ' ' {
                            chars.push(c);
                        }
                    }
                    chars.extend(" SPRITE".chars());
                    let mut addr =
                        snes2pc(0xceb240 + (236 - 128) * 0x40) + 0x20 - chars.len().div_ceil(2) * 2;
                    for c in chars {
                        let color_palette = 0x0400;
                        if c.is_ascii_uppercase() {
                            rom.write_u16(addr, (c as isize - 'A' as isize) | color_palette)?;
                        }
                        addr += 2;
                    }

                    // Write the sprite author
                    let mut chars = vec![];
                    let author = info.authors.join(", ");
                    for c in author.chars() {
                        let c = c.to_ascii_uppercase();
                        if c.is_ascii_uppercase() || c.is_ascii_digit() || c == ' ' {
                            chars.push(c);
                        }
                    }
                    let mut addr =
                        snes2pc(0xceb240 + (237 - 128) * 0x40) + 0x20 - chars.len().div_ceil(2) * 2;
                    for c in chars {
                        write_credits_big_char(rom, c, addr)?;
                        addr += 2;
                    }
                }
            }
        }
    }

    Ok(())
}

pub fn parse_controller_button(s: &str) -> Result<ControllerButton> {
    Ok(match s {
        "Left" => ControllerButton::Left,
        "Right" => ControllerButton::Right,
        "Up" => ControllerButton::Up,
        "Down" => ControllerButton::Down,
        "X" => ControllerButton::X,
        "Y" => ControllerButton::Y,
        "A" => ControllerButton::A,
        "B" => ControllerButton::B,
        "Select" => ControllerButton::Select,
        "L" => ControllerButton::L,
        "R" => ControllerButton::R,
        _ => bail!("Unexpected controller button: {}", s),
    })
}

fn get_button_mask(mut controller_button: ControllerButton, default: ControllerButton) -> isize {
    if let ControllerButton::Default = controller_button {
        controller_button = default;
    }
    match controller_button {
        ControllerButton::Left => 0x0200,
        ControllerButton::Right => 0x0100,
        ControllerButton::Up => 0x0800,
        ControllerButton::Down => 0x0400,
        ControllerButton::X => 0x0040,
        ControllerButton::Y => 0x4000,
        ControllerButton::A => 0x0080,
        ControllerButton::B => 0x8000,
        ControllerButton::L => 0x0020,
        ControllerButton::R => 0x0010,
        ControllerButton::Select => 0x2000,
        ControllerButton::Start => 0x1000,
        _ => panic!("Unexpected controller button: {controller_button:?}"),
    }
}

fn get_button_list_mask(buttons: &[ControllerButton]) -> isize {
    let mut mask = 0x0000;
    for &button in buttons {
        mask |= get_button_mask(button, ControllerButton::Default);
    }
    if mask == 0x0000 {
        // If no button are specified, assume this input combination (e.g. quick reload or spin lock)
        // is disabled, rather than being activated with no inputs held.
        mask = 0xFFFF;
    }
    mask
}

fn apply_controller_config(rom: &mut Rom, controller_config: &ControllerConfig) -> Result<()> {
    let control_data = vec![
        (0x81B325, controller_config.jump, ControllerButton::A),
        (0x81B32B, controller_config.dash, ControllerButton::B),
        (0x81B331, controller_config.shot, ControllerButton::X),
        (0x81B337, controller_config.item_cancel, ControllerButton::Y),
        (
            0x81B33D,
            controller_config.item_select,
            ControllerButton::Select,
        ),
        (0x81B343, controller_config.angle_up, ControllerButton::R),
        (0x81B349, controller_config.angle_down, ControllerButton::L),
    ];
    for (addr, button, default) in control_data {
        let mask = get_button_mask(button, default);
        rom.write_u16(snes2pc(addr), mask)?;
    }

    let spin_lock_mask = get_button_list_mask(&controller_config.spin_lock_buttons);
    rom.write_u16(snes2pc(0x82FE7C), spin_lock_mask)?;

    let quick_reload_mask = get_button_list_mask(&controller_config.quick_reload_buttons);
    rom.write_u16(snes2pc(0x82FE7E), quick_reload_mask)?;

    if controller_config.moonwalk {
        apply_ips_patch(rom, Path::new("../patches/ips/enable_moonwalk.ips"))?;
    }
    // $82FE7E

    Ok(())
}

fn disable_songset(
    rom: &mut Rom,
    block_spc: usize,
    block_pc: usize,
    block_size: usize,
    songset_idx: usize,
    songset_size: usize,
) -> Result<()> {
    let spc2pc = |x: usize| {
        assert!(x >= block_spc);
        assert!(x < block_spc + block_size);
        x - block_spc + block_pc
    };

    for i in 0..songset_size {
        let tracker_ptr = rom.read_u16(spc2pc(0x5828 + 2 * i))? as usize;
        let mut addr = tracker_ptr;
        loop {
            let cmd = rom.read_u16(spc2pc(addr))? as usize;
            addr += 2;
            if cmd == 0x0000 {
                break;
            }
            if cmd == 0x00ff {
                addr += 2;
                continue;
            }
            info!(
                "songset {} ({}+{}), tracker_ptr {:x}, cmd {:x}",
                songset_idx, block_spc, block_size, tracker_ptr, cmd
            );
            for j in 0..8 {
                let track_ptr = rom.read_u16(spc2pc(cmd + 2 * j))? as usize;
                if track_ptr == 0 {
                    continue;
                }
                info!("track_ptr: {:x}", track_ptr);
                rom.write_n(spc2pc(track_ptr), &[0xc9, 0x00])?;
            }
        }
    }

    Ok(())
}

fn disable_music(rom: &mut Rom) -> Result<()> {
    // Old way of disabling the music, which didn't allow flexibility
    // to keep certain tracks enabled:
    // rom.write_u8(snes2pc(0xcf8413), 0x6F)?;

    // Shared music tracks, defined as part of the SPC engine loaded at bootup:
    #[rustfmt::skip]
    let shared_tracks = [
        // samus appears (keep enabled):
        // 0x5322, 0x535D, 0x53AF, 0x53C6, 0x53E0, 0x53F5, 0x5431,
        // item fanfare (keep enabled):
        // 0x5482, 0x54AA, 0x54BD, 0x54D8, 0x54EB, 0x550B, 0x553B,
        // elevator music 1:
        0x5593, 0x55C8, 0x55EE, 0x5630, 
        // elevator music 2:
        0x5649, 0x5665, 0x566D, 0x5675, 0x567D, 
        // 0x56AF  // statues hallways (keep enabled)
    ];

    for ptr in shared_tracks {
        let addr = snes2pc(0xcf8108) + ptr - 0x1500;
        rom.write_n(addr, &[0xc9, 0x00])?;
    }

    let songset_sizes: Vec<(usize, usize)> = vec![
        (0, 3),  // $CF:8000: SPC engine
        (1, 2),  // $D0:E20D: Title sequence
        (2, 3),  // $D1:B62A: Empty Crateria
        (3, 2),  // $D2:88CA: Lower Crateria
        (4, 1),  // $D2:D9B6: Upper Crateria
        (5, 1),  // $D3:933C: Green Brinstar
        (6, 1),  // $D3:E812: Red Brinstar
        (7, 1),  // $D4:B86C: Upper Norfair
        (8, 1),  // $D4:F420: Lower Norfair
        (9, 2),  // $D5:C844: Maridia
        (10, 2), // $D6:98B7: Tourian
        (11, 1), // $D6:EF9D: Mother Brain
        (12, 3), // $D7:BF73: Boss fight 1
        (13, 2), // $D8:99B2: Boss fight 2
        (14, 1), // $D8:EA8B: Miniboss fight
        (15, 4), // $D9:B67B: Ceres
        (16, 2), // $D9:F5DD: Wrecked Ship
        // (17, 1), // $DA:B650: Zebes boom (keep enabled)
        (18, 1), // $DA:D63B: Intro
        // (19, 1), // $DB:A40F: Death (keep enabled)
        (20, 1), // $DB:DF4F: Credits
        // (21, 1), // $DC:AF6C: "The last Metroid is in captivity" (unused)
        // (22, 1), // $DC:FAC7: "The galaxy is at peace" (unused)
        (23, 3), // $DD:B104: Big Boy (same as boss fight 2)
        (24, 1), // $DE:81C1: Samus theme (same as upper Crateria)
    ];

    for (songset_idx, songset_size) in songset_sizes {
        let songset_ptr = snes2pc(rom.read_u24(snes2pc(0x8fe7e1) + songset_idx * 3)? as usize);
        let mut addr = songset_ptr;
        loop {
            let size = rom.read_u16(addr)? as usize;
            if size == 0 {
                break;
            }
            let dst = rom.read_u16(addr + 2)? as usize;

            if (dst..(dst + size)).contains(&0x5828) {
                disable_songset(rom, dst, addr + 4, size, songset_idx, songset_size)?;
            }
            addr += 4 + size;
        }
    }

    Ok(())
}

pub fn customize_rom(
    rom: &mut Rom,
    orig_rom: &Rom,
    map: &Map,
    settings: &CustomizeSettings,
    game_data: &GameData,
    samus_sprite_categories: &[SamusSpriteCategory],
    mosaic_themes: &[MosaicTheme],
) -> Result<()> {
    remove_mother_brain_flashing(rom)?;
    apply_retiling(
        rom,
        orig_rom,
        map,
        game_data,
        &settings.tile_theme,
        settings.statues_hallway_tiling,
        mosaic_themes,
    )?;

    match &settings.palette_theme {
        PaletteTheme::Vanilla => {}
        PaletteTheme::AreaThemed => {
            apply_area_themed_palettes(rom)?;
        }
    }

    match settings.door_theme {
        DoorTheme::Vanilla => {}
        DoorTheme::Alternate => {
            apply_ips_patch(rom, Path::new("../patches/ips/alternate_door_colors.ips"))?;
        }
    }

    let mut map_icon_settings = 0;
    if !settings.boss_icons {
        map_icon_settings |= 1;
    }
    if !settings.miniboss_icons {
        map_icon_settings |= 2;
    }
    if !settings.save_icons {
        map_icon_settings |= 4;
    }
    rom.write_u16(snes2pc(0x85B600), map_icon_settings)?;

    // Fix Phantoon power-on sequence to not overwrite the first two palettes, since those contain
    // customized HUD colors which would get messed up.
    rom.write_u16(snes2pc(0xA7DC6E), 0x0040)?;

    apply_custom_samus_sprite(rom, settings, samus_sprite_categories)?;
    if let Some((r, g, b)) = settings.etank_color {
        let color = (r as isize) | ((g as isize) << 5) | ((b as isize) << 10);
        rom.write_u16(snes2pc(0x82FFFE), color)?; // Gameplay ETank color
        rom.write_u16(snes2pc(0x8EE416), color)?; // Main menu
        rom.write_u16(snes2pc(0xA7CA7B), color)?; // During Phantoon power-on
    }
    if settings.reserve_hud_style {
        apply_ips_patch(rom, Path::new("../patches/ips/reserve_hud.ips"))?;
    }
    if settings.room_names {
        rom.write_u16(snes2pc(0x82FFFA), 1)?;
    } else {
        rom.write_u16(snes2pc(0x82FFFA), 0)?;
    }

    if settings.statues_hallway_audio == StatuesHallwayAudio::Louder {
        // Duplicate the Statues Hallway track across multiple channels, to make it more audible:
        // (This could be increased to up to 8 channels, to make it even more loud.)
        for i in 0..4 {
            rom.write_u16(snes2pc(0xcf8108) + 0x569f - 0x1500 + i * 2, 0x56af)?;
        }
    }

    match settings.music {
        MusicSettings::AreaThemed => {}
        MusicSettings::Disabled => {
            // We could call `override_music` here to restore the vanilla tracks: this would restore the correct sound effects
            // but at a cost of increasing room load times by almost 1 second per room.
            // override_music(rom)?;
            disable_music(rom)?;
        }
    }
    if settings.disable_beeping {
        rom.write_n(snes2pc(0x90EA92), &[0xEA; 4])?;
        rom.write_n(snes2pc(0x90EAA0), &[0xEA; 4])?;
        rom.write_n(snes2pc(0x90F33C), &[0xEA; 4])?;
        rom.write_n(snes2pc(0x91E6DA), &[0xEA; 4])?;
    }
    match settings.shaking {
        ShakingSetting::Vanilla => {}
        ShakingSetting::Reduced => {
            // Limit BG shaking to 1-pixel displacements:
            for i in 0..144 {
                let x = rom.read_u16(snes2pc(0xA0872D + i * 2))?;
                rom.write_u16(snes2pc(0xA0872D + i * 2), min(x, 1))?;
            }
            // (Enemies already only shake up to 1 pixel)
            // Limit enemy projectile shaking to 1-pixel displacements:
            for i in 0..72 {
                let x = rom.read_u16(snes2pc(0x86846B + i * 2))?;
                rom.write_u16(snes2pc(0x86846B + i * 2), min(x, 1))?;
            }
        }
        ShakingSetting::Disabled => {
            // Disable BG shaking globally, by setting the shake displacements to zero (this should be timing-neutral?)
            rom.write_n(snes2pc(0xA0872D), &[0; 288])?;
            // Disable enemy shaking:
            rom.write_n(snes2pc(0xA09488), &[0xEA; 5])?; // 5 * NOP
            rom.write_n(snes2pc(0xA0948F), &[0xEA; 5])?; // 5 * NOP

            // Disable enemy projectile shaking, by setting the displacements to zero:
            rom.write_n(snes2pc(0x86846B), &[0; 144])?;
        }
    }
    match settings.flashing {
        FlashingSetting::Vanilla => {
            apply_ips_patch(rom, Path::new("../patches/ips/flashing_placebo.ips"))?;
        }
        FlashingSetting::Reduced => {
            apply_ips_patch(rom, Path::new("../patches/ips/flashing_placebo.ips"))?;
            write_glowpatch(rom, &game_data.reduced_flashing_patch)?;
        }
    }
    apply_controller_config(rom, &settings.controller_config)?;
    Ok(())
}
