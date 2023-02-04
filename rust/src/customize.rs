mod room_palettes;

use room_palettes::apply_area_themed_palettes;
use crate::{
    game_data::GameData,
    patch::Rom,
};
use anyhow::Result;

pub struct CustomizeSettings {
    pub area_themed_palette: bool,
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
    if settings.area_themed_palette {
        apply_area_themed_palettes(rom, game_data)?;
    }
    Ok(())
}
