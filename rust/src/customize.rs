pub mod retiling;
pub mod room_palettes;
pub mod vanilla_music;

use std::cmp::min;
use anyhow::{bail, Result};
use std::path::Path;

use crate::customize::vanilla_music::override_music;
use crate::{
    game_data::GameData,
    patch::{apply_ips_patch, snes2pc, write_credits_big_char, Rom},
    web::SamusSpriteCategory,
};
use retiling::apply_retiling;
use room_palettes::apply_area_themed_palettes;

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
                // println!("success: allocated {} bytes: ending at {:x}", size, pc2snes(block.current_addr));
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
    Vanilla,
    AreaThemed,
    Disabled,
}

#[derive(Debug)]
pub enum AreaTheming {
    Vanilla,
    Palettes,
    Tiles(String),
}

#[derive(Default, Debug, Copy, Clone)]
pub enum ControllerButton {
    #[default]
    Default,
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
    pub quick_reload_buttons: Vec<ControllerButton>,
    pub moonwalk: bool,
}

#[derive(Debug, Copy, Clone)]
pub enum ShakingSetting {
    Vanilla,
    Reduced,
    Disabled
}

#[derive(Debug)]
pub struct CustomizeSettings {
    pub samus_sprite: Option<String>,
    pub etank_color: Option<(u8, u8, u8)>,
    pub reserve_hud_style: bool,
    pub vanilla_screw_attack_animation: bool,
    pub area_theming: AreaTheming,
    pub music: MusicSettings,
    pub disable_beeping: bool,
    pub shaking: ShakingSetting,
    pub controller_config: ControllerConfig,
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
        let sprite_name = settings.samus_sprite.clone().unwrap_or("samus".to_string());
        let patch_path_str = format!("../patches/samus_sprites/{}.ips", sprite_name);
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
                        if (c >= 'A' && c <= 'Z') || c == ' ' {
                            chars.push(c);
                        }
                    }
                    chars.extend(" SPRITE".chars());
                    let mut addr =
                        snes2pc(0xceb240 + (234 - 128) * 0x40) + 0x20 - (chars.len() + 1) / 2 * 2;
                    for c in chars {
                        let color_palette = 0x0400;
                        if c >= 'A' && c <= 'Z' {
                            rom.write_u16(addr, (c as isize - 'A' as isize) | color_palette)?;
                        }
                        addr += 2;
                    }

                    // Write the sprite author
                    let mut chars = vec![];
                    let author = info.authors.join(", ");
                    for c in author.chars() {
                        let c = c.to_ascii_uppercase();
                        if (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == ' ' {
                            chars.push(c);
                        }
                    }
                    let mut addr =
                        snes2pc(0xceb240 + (235 - 128) * 0x40) + 0x20 - (chars.len() + 1) / 2 * 2;
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
        "X" => ControllerButton::X,
        "Y" => ControllerButton::Y,
        "A" => ControllerButton::A,
        "B" => ControllerButton::B,
        "Select" => ControllerButton::Select,
        "L" => ControllerButton::L,
        "R" => ControllerButton::R,
        _ => bail!("Unexpected controller button: {}", s)
    })
}

fn get_button_mask(mut controller_button: ControllerButton, default: ControllerButton) -> isize {
    if let ControllerButton::Default = controller_button {
        controller_button = default;
    }
    match controller_button {
        ControllerButton::X => 0x0040,
        ControllerButton::Y => 0x4000,
        ControllerButton::A => 0x0080,
        ControllerButton::B => 0x8000,
        ControllerButton::L => 0x0020,
        ControllerButton::R => 0x0010,
        ControllerButton::Select => 0x2000,
        ControllerButton::Start => 0x1000,
        _ => panic!("Unexpected controller button: {:?}", controller_button)
    }
}

fn apply_controller_config(rom: &mut Rom, controller_config: &ControllerConfig) -> Result<()> {
    let control_data = vec![
        (0x81B325, controller_config.jump, ControllerButton::A),
        (0x81B32B, controller_config.dash, ControllerButton::B),
        (0x81B331, controller_config.shot, ControllerButton::X),
        (0x81B337, controller_config.item_cancel, ControllerButton::Y),
        (0x81B33D, controller_config.item_select, ControllerButton::Select),
        (0x81B343, controller_config.angle_up, ControllerButton::R),
        (0x81B349, controller_config.angle_down, ControllerButton::L),
    ];
    for (addr, button, default) in control_data {
        let mask = get_button_mask(button, default);
        rom.write_u16(snes2pc(addr), mask)?;
    }
    
    let mut quick_reload_mask = 0x0000;
    for &button in &controller_config.quick_reload_buttons {
        quick_reload_mask |= get_button_mask(button, ControllerButton::Default);
    }
    if quick_reload_mask == 0x0000 {
        // The user probably intended to disable quick-reload entirely (rather than having quick reload trigger
        // when not holding any buttons), so that's what we do, effectively, by requiring all 12 buttons to be pressed:
        quick_reload_mask = 0xFFFF;
    }
    rom.write_u16(snes2pc(0x82FE7E), quick_reload_mask)?;

    if controller_config.moonwalk {
        apply_ips_patch(rom, Path::new("../patches/ips/enable_moonwalk.ips"))?;
    }
    // $82FE7E

    Ok(())
}

pub fn customize_rom(
    rom: &mut Rom,
    orig_rom: &Rom,
    seed_patch: &[u8],
    settings: &CustomizeSettings,
    game_data: &GameData,
    samus_sprite_categories: &[SamusSpriteCategory],
) -> Result<()> {
    rom.resize(0x400000);
    let patch = ips::Patch::parse(seed_patch).unwrap();
    // .with_context(|| format!("Unable to parse patch {}", patch_path.display()))?;
    for hunk in patch.hunks() {
        rom.write_n(hunk.offset(), hunk.payload())?;
    }

    remove_mother_brain_flashing(rom)?;
    match &settings.area_theming {
        AreaTheming::Vanilla => {}
        AreaTheming::Palettes => {
            apply_area_themed_palettes(rom, game_data)?;
        }
        AreaTheming::Tiles(theme) => {
            apply_retiling(rom, orig_rom, game_data, &theme)?;
            // // Failed attempt to put Dachora further back, e.g. so it doesn't go in front of Crateria tube:
            // rom.write_u8(snes2pc(0xA0E5FF + 0x39), 0x06)?;
        }
    }
    apply_custom_samus_sprite(rom, settings, samus_sprite_categories)?;
    if let Some((r, g, b)) = settings.etank_color {
        let color = (r as isize) | ((g as isize) << 5) | ((b as isize) << 10);
        rom.write_u16(snes2pc(0x82FFFE), color)?; // Gameplay ETank color
                                                  // rom.write_u16(snes2pc(0xB6F01A), color)?;
        rom.write_u16(snes2pc(0x8EE416), color)?; // Main menu
        rom.write_u16(snes2pc(0xA7CA7B), color)?; // During Phantoon power-on
    }
    if settings.reserve_hud_style {
        apply_ips_patch(rom, Path::new("../patches/ips/reserve_hud.ips"))?;
        // Make used reserve tiles empty, for when they appear when transitioning to and from Kraid's room
        // Since the current IPS creation tool doesn't include settings these addresses to zero, it has to be done here instead
        for i in 0..6 {
            rom.write_n(snes2pc(0xE20000+(0x10000*i)+0xC330), &[0x00; 0x10])?;
            rom.write_n(snes2pc(0xE20000+(0x10000*i)+0xC460), &[0x00; 0x10])?;
            rom.write_n(snes2pc(0xE20000+(0x10000*i)+0xC4C0), &[0x00; 0x20])?;
        }
    }
    match settings.music {
        MusicSettings::Vanilla => {
            override_music(rom)?;
        }
        MusicSettings::AreaThemed => {}
        MusicSettings::Disabled => {
            override_music(rom)?;
            rom.write_u8(snes2pc(0xcf8413), 0x6F)?;
        }
    }
    if settings.disable_beeping {
        rom.write_n(snes2pc(0x90EA92), &[0xEA; 4])?;
        rom.write_n(snes2pc(0x90EAA0), &[0xEA; 4])?;
        rom.write_n(snes2pc(0x90F33C), &[0xEA; 4])?;
        rom.write_n(snes2pc(0x91E6DA), &[0xEA; 4])?;
    }
    match settings.shaking {
        ShakingSetting::Vanilla => {},
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
        },
        ShakingSetting::Disabled => {
            // Disable BG shaking globally, by setting the shake displacements to zero (this should be timing-neutral?)
            rom.write_n(snes2pc(0xA0872D), &[0; 288])?;
            // Disable enemy shaking:
            rom.write_n(snes2pc(0xA09488), &[0xEA; 5])?;  // 5 * NOP
            rom.write_n(snes2pc(0xA0948F), &[0xEA; 5])?;  // 5 * NOP
            // rom.write_u8(snes2pc(0xA08712), 0x60)?;  // RTS

            // Disable enemy projectile shaking, by setting the displacements to zero:
            rom.write_n(snes2pc(0x86846B), &[0; 144])?;
        }
    }
    apply_controller_config(rom, &settings.controller_config)?;
    Ok(())
}
