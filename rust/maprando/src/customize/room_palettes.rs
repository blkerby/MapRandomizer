use crate::patch::{Rom, snes2pc};
use anyhow::Result;

pub fn encode_palette(pal: &[[u8; 3]]) -> Vec<u8> {
    let mut out: Vec<u8> = vec![];
    for c in pal {
        let r = c[0] as u16 / 8;
        let g = c[1] as u16 / 8;
        let b = c[2] as u16 / 8;
        let w = r | (g << 5) | (b << 10);
        out.push((w & 0xFF) as u8);
        out.push((w >> 8) as u8);
    }
    out
}

pub fn decode_palette(pal_bytes: &[u8]) -> [[u8; 3]; 128] {
    let mut out = [[0u8; 3]; 128];
    for i in 0..128 {
        let c = pal_bytes[i * 2] as u16 | ((pal_bytes[i * 2 + 1] as u16) << 8);
        let r = (c & 31) * 8;
        let g = ((c >> 5) & 31) * 8;
        let b = ((c >> 10) & 31) * 8;
        out[i] = [r as u8, g as u8, b as u8];
    }
    out
}

fn lighten_firefleas(rom: &mut Rom) -> Result<()> {
    // Reduce the darkening effect per fireflea kill (so that in many of the palettes the
    // room won't go completely black so soon).
    let darkness_shades = [
        0x00, 0x00, 0x00, 0x03, 0x00, 0x06, 0x00, 0x0A, 0x00, 0x10, 0x00, 0x12,
    ];
    rom.write_n(snes2pc(0x88B070), &darkness_shades)?;
    Ok(())
}

pub fn apply_area_themed_palettes(rom: &mut Rom) -> Result<()> {
    // Set flag to enable behavior in "Area Palettes.asm":
    rom.write_u16(snes2pc(0x8AC000), 0xF0F0)?;

    lighten_firefleas(rom)?;
    Ok(())
}
