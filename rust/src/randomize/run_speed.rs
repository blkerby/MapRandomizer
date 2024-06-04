// Runway distance travelled (in subpixels) by frame, with dash held the whole time,
// with SpeedBooster equipped:
const RUN_SPEED_TABLE: [i32; 112] = [
    0x14,
    0x1C,
    0x28,
    0x38,
    0x4C,
    0x64,
    0x80,
    0xA0,
    0xC4,
    0xEC,
    0x123,
    0x15B,
    0x194,
    0x1CE,
    0x209,
    0x245,
    0x282,
    0x2C0,
    0x2FF,
    0x33F,
    0x380,
    0x3C2,
    0x405,
    0x449,
    0x48E,
    0x4D4,
    0x51B,
    0x563,
    0x5AC,
    0x5F6,
    0x641,
    0x68D,
    0x6DA,
    0x728,
    0x777,
    0x7C7,
    0x818,
    0x86A,
    0x8BD,
    0x911,
    0x966,
    0x9BC,
    0xA13,
    0xA6B,
    0xAC4,
    0xB1E,
    0xB79,
    0xBD5,
    0xC32,
    0xC90,
    0xCEF,
    0xD4F,
    0xDB0,
    0xE12,
    0xE75,
    0xED9,
    0xF3E,
    0xFA4,
    0x100B,
    0x1073,
    0x10DC,
    0x1146,
    0x11B1,
    0x121D,
    0x128A,
    0x12F8,
    0x1367,
    0x13D7,
    0x1448,
    0x14BA,
    0x152D,
    0x15A1,
    0x1616,
    0x168C,
    0x1703,
    0x177B,
    0x17F4,
    0x186E,
    0x18E9,
    0x1965,
    0x19E2,
    0x1A60,
    0x1ADF,
    0x1B5F,
    0x1BE0,
    0x1C62,
    0x1CE5,
    0x1D69,
    0x1DEE,
    0x1E74,
    0x1EFB,
    0x1F83,
    0x200C,
    0x2096,
    0x2121,
    0x21AD,
    0x223A,
    0x22C8,
    0x2357,
    0x23E7,
    0x2478,
    0x250A,
    0x259D,
    0x2631,
    0x26C6,
    0x275C,
    0x27F3,
    0x288B,
    0x2924,
    0x29BE,
    0x2A59,
    0x2AF5,
];

fn linear_interpolate(x: f32, table: &[(i32, i32)]) -> f32 {
    if x <= table[0].0 as f32 {
        return table[0].1 as f32;
    }
    if x >= table.last().unwrap().0 as f32 {
        return table.last().unwrap().1 as f32;
    }
    let i = match table.binary_search_by_key(&(x as i32), |(x, y)| *x) {
        Ok(i) => i,
        Err(i) => i - 1,
    };
    let x0 = table[i].0 as f32;
    let x1 = table[i + 1].0 as f32;
    let y0 = table[i].1 as f32;
    let y1 = table[i + 1].1 as f32;
    (x as f32 - x0) / (x1 - x0) * (y1 - y0) + y0
}

// Maximum extra run speed (in pixels per frame) obtainable by running on a given length of runway
// and jumping before the end of it, with SpeedBooster equipped.
pub fn get_max_extra_run_speed(runway_length: f32) -> f32 {
    let runway_subpixels = (runway_length * 16.0) as i32;
    match RUN_SPEED_TABLE.binary_search(&runway_subpixels) {
        Ok(i) => (i as f32) / 16.0,
        Err(i) => (i as f32) / 16.0,
    }
}

// Minimum extra run speed (in pixels per frame) obtainable by gaining a shortcharge
// at the given skill level (in minimum number of tiles to gain a shortcharge)
pub fn get_shortcharge_min_extra_run_speed(shortcharge_tile_skill: f32) -> f32 {
    // Table mapping minimum shortcharge tiles into number of frames with dash held:
    let table: Vec<(i32, i32)> = vec![
        (11, 0x07),
        (12, 0x0B),
        (13, 0x10),
        (14, 0x12),
        (15, 0x16),
        (16, 0x1B),
        (17, 0x1E),
        (20, 0x37),
        (25, 0x48),
        (30, 0x59),
    ];
    linear_interpolate(shortcharge_tile_skill, &table)
}

pub fn get_shortcharge_max_extra_run_speed(shortcharge_tile_skill: f32, runway_length: f32) -> Option<f32> {
    if shortcharge_tile_skill > runway_length {
        return None;
    }
    if runway_length >= 30.0 {
        return Some(get_max_extra_run_speed(runway_length));
    }
    // Table of maximum run speed obtainable at given shortcharge skill level, at specific runway lengths:
    let table = if shortcharge_tile_skill >= 25.0 {
        vec![
            (25, 0x48),
            (30, 0x59),
        ]
    } else if shortcharge_tile_skill >= 20.0 {
        vec![
            (20, 0x3A),
            (25, 0x4B),
            (30, 0x59),
        ]
    } else if shortcharge_tile_skill >= 16.0 {
        vec![
            (16, 0x23),
            (17, 0x29),
            (20, 0x3C),
            (25, 0x4C),
            (30, 0x59),
        ]
    } else if shortcharge_tile_skill >= 15.0 {
        vec![
            (15, 0x1B),
            (16, 0x2B),
            (17, 0x2D),
            (20, 0x3E),
            (25, 0x4D),
            (30, 0x59),
        ]
    } else if shortcharge_tile_skill >= 14.0 {
        vec![
            (14, 0x1A),
            (15, 0x29),
            (16, 0x31),
            (17, 0x33),
            (20, 0x40),
            (25, 0x4E),
            (30, 0x59),
        ]
    } else if shortcharge_tile_skill >= 13.0 {
        vec![
            (13, 0x12),
            (14, 0x24),
            (15, 0x2B),
            (16, 0x33),
            (17, 0x38),
            (20, 0x41),
            (25, 0x4E),
            (30, 0x59),
        ]
    } else {
        vec![
            (11, 0x07),
            (13, 0x0B),
            (14, 0x2A),
            (15, 0x2F),
            (16, 0x35),
            (17, 0x3A),
            (20, 0x42),
            (25, 0x4E),
            (30, 0x59),
        ]
    };
    Some(linear_interpolate(runway_length, &table) / 16.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_interpolate() {
        let table = vec![
            (3, 10),
            (5, 16),
            (9, 20),
        ];
        assert_eq!(linear_interpolate(0.0, &table), 10.0);
        assert_eq!(linear_interpolate(3.0, &table), 10.0);
        assert_eq!(linear_interpolate(4.0, &table), 13.0);
        assert_eq!(linear_interpolate(5.0, &table), 16.0);
        assert_eq!(linear_interpolate(6.0, &table), 17.0);
        assert_eq!(linear_interpolate(9.0, &table), 20.0);
        assert_eq!(linear_interpolate(10.0, &table), 20.0);
    }
}