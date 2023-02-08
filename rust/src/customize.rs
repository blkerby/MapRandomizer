mod room_palettes;

use room_palettes::apply_area_themed_palettes;
use crate::{
    game_data::GameData,
    patch::{Rom, snes2pc},
};
use anyhow::Result;

#[derive(Debug)]
pub struct CustomizeSettings {
    pub area_themed_palette: bool,
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
) -> Result<()> {
    let patch = ips::Patch::parse(seed_patch).unwrap();
    // .with_context(|| format!("Unable to parse patch {}", patch_path.display()))?;
    for hunk in patch.hunks() {
        rom.write_n(hunk.offset(), hunk.payload())?;
    }
    remove_mother_brain_flashing(rom)?;
    if settings.area_themed_palette {
        apply_area_themed_palettes(rom, game_data)?;
    }
    Ok(())
}
