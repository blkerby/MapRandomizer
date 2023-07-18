mod room_palettes;

use std::path::Path;

use room_palettes::apply_area_themed_palettes;
use crate::{
    game_data::GameData,
    patch::{Rom, snes2pc, apply_ips_patch, write_credits_big_char}, web::{SamusSpriteInfo, SamusSpriteCategory},
};
use anyhow::Result;

#[derive(Debug)]
pub enum MusicSettings {
    Vanilla,
    AreaThemed,
    Disabled
}

#[derive(Debug)]
pub struct CustomizeSettings {
    pub samus_sprite: Option<String>,
    pub vanilla_screw_attack_animation: bool,
    pub area_themed_palette: bool,
    pub music: MusicSettings,
    pub disable_beeping: bool,
}

fn remove_mother_brain_flashing(rom: &mut Rom) -> Result<()> {
    // Disable start of flashing after Mother Brain 1:
    rom.write_u16(snes2pc(0xA9CFFE), 0)?;

    // Disable end of flashing (to prevent palette from getting overwritten)
    rom.write_u8(snes2pc(0xA9D00C), 0x60)?;  // RTS

    Ok(())
}


pub fn customize_rom(
    rom: &mut Rom,
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
                    let credits_name = info.credits_name.clone().unwrap_or(info.display_name.clone());
                    for c in credits_name.chars() {
                        let c = c.to_ascii_uppercase();
                        if (c >= 'A' && c <= 'Z') || c == ' ' {
                            chars.push(c);
                        }
                    }
                    chars.extend(" SPRITE".chars());
                    let mut addr = snes2pc(0xceb240 + (234 - 128) * 0x40) + 0x20 - (chars.len() + 1) / 2 * 2;
                    for c in chars {
                        let color_palette = 0x0400;
                        if c >= 'A' && c <= 'Z' {
                            rom.write_u16(addr, (c as isize - 'A' as isize) | color_palette)?;
                        }
                        addr += 2;
                    }

                    // Write the sprite author
                    let mut chars = vec![];
                    for c in info.author.chars() {
                        let c = c.to_ascii_uppercase();
                        if (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == ' ' {
                            chars.push(c);
                        }
                    }
                    let mut addr = snes2pc(0xceb240 + (235 - 128) * 0x40) + 0x20 - (chars.len() + 1) / 2 * 2;
                    for c in chars {
                        write_credits_big_char(rom, c, addr)?;
                        addr += 2;
                    }
                }
            }
        }
    }

    remove_mother_brain_flashing(rom)?;
    if settings.area_themed_palette {
        apply_area_themed_palettes(rom, game_data)?;
    }
    match settings.music {
        MusicSettings::Vanilla => {
            apply_ips_patch(rom, Path::new(&"../patches/ips/music.ips"))?;
        }
        MusicSettings::AreaThemed => {}
        MusicSettings::Disabled => {
            rom.write_u8(snes2pc(0xcf8413), 0x6F)?;
        }
    }
    if settings.disable_beeping {
        rom.write_n(snes2pc(0x90EA92), &[0xEA; 4])?;
        rom.write_n(snes2pc(0x90EAA0), &[0xEA; 4])?;
        rom.write_n(snes2pc(0x90F33C), &[0xEA; 4])?;
        rom.write_n(snes2pc(0x91E6DA), &[0xEA; 4])?;
    }
    Ok(())
}
