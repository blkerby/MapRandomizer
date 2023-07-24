// In the vanilla game, many rooms have their songset and/or play index as "no change", so they depend on surrounding
// rooms to start the correct song. This would give unexpected results when we rearrange the rooms, so we replace
// "no change" with a specific songset & play-index (usually the one from the vanilla game).
//
// We're going to be overwriting a ROM with area-themed music, so we need to overwrite the song in every room/state.

use anyhow::Result;

use crate::patch::Rom;

pub fn override_music(rom: &mut Rom) -> Result<()> {
    let song_overrides = vec![
        // Crateria
        (0x9217, 0x06, 0x05), // room $91F8 - Landing Site - state: Zebes not awake
        (0x9231, 0x06, 0x06), // room $91F8 - Landing Site - state: Zebes awake
        (0x924B, 0x0C, 0x05), // room $91F8 - Landing Site - state: Power Bombs
        (0x9265, 0x00, 0x00), // room $91F8 - Landing Site - state: escape
        (0x92C9, 0x06, 0x05), // room $92B3 - Gauntlet Entrance - state: Zebes not awake
        (0x92E3, 0x09, 0x05), // room $92B3 - Gauntlet Entrance - state: Zebes awake
        (0x9318, 0x06, 0x05), // room $92FD - Parlor and Alcatraz - state: Zebes not awake
        (0x9332, 0x09, 0x05), // room $92FD - Parlor and Alcatraz - state: Zebes awake
        (0x934C, 0x00, 0x00), // room $92FD - Parlor and Alcatraz - state: escape (This doesn't matter)
        (0x93BB, 0x09, 0x05), // room $93AA - Crateria Power Bomb Room
        (0x93E6, 0x09, 0x05), // room $93D5 - Crateria Save Room
        (0x940F, 0x0C, 0x05), // room $93FE - West Ocean
        (0x9472, 0x0C, 0x05), // room $9461 - Bowling Alley Path
        (0x949D, 0x09, 0x05), // room $948C - Crateria Kihunter Room
        (0x94DD, 0x00, 0x03), // room $94CC - Forgotten Highway Elevator
        (0x950E, 0x0C, 0x05), // room $94FD - East Ocean
        (0x9563, 0x0C, 0x05), // room $9552 - Forgotten Highway Kago Room
        (0x958E, 0x0C, 0x05), // room $957D - Crab Maze
        (0x95B9, 0x0C, 0x05), // room $95A8 - Forgotten Highway Elbow
        (0x95E5, 0x09, 0x05), // room $95D4 - Crateria Tube
        (0x9610, 0x09, 0x05), // room $95FF - The Moat
        (0x963B, 0x00, 0x03), // room $962A - Red Brinstar Elevator Room
        (0x966C, 0x09, 0x05), // room $965B - Gauntlet Energy Tank Room
        (0x96A0, 0x0C, 0x05), // room $968F - Homing Geemer Room
        (0x96D5, 0x06, 0x05), // room $96BA - Climb - state: Zebes not awake
        (0x96EF, 0x09, 0x05), // room $96BA - Climb - state: Zebes awake
        (0x9709, 0x00, 0x00), // room $96BA - Climb - state: escape (This doesn't matter)
        (0x9771, 0x06, 0x05), // room $975C - Pit Room - state: Zebes not awake
        (0x978B, 0x09, 0x05), // room $975C - Pit Room - state: Zebes awake
        (0x97CA, 0x06, 0x05), // room $97B5 - Blue Brinstar Elevator Room - state: Zebes not awake
        (0x97E4, 0x00, 0x03), // room $97B5 - Blue Brinstar Elevator Room - state: Zebes awake
        (0x981F, 0x24, 0x03), // room $9804 - Bomb Torizo Room - state: torizo not dead
        (0x9839, 0x00, 0x03), // room $9804 - Bomb Torizo Room - state: torizo dead
        (0x9853, 0x00, 0x00), // room $9804 - Bomb Torizo Room - state: escape (This doesn't matter)
        (0x9894, 0x09, 0x05), // room $9879 - Flyway - state: torizo not dead
        (0x98AE, 0x09, 0x05), // room $9879 - Flyway - state: torizo dead
        (0x98C8, 0x24, 0x07), // room $9879 - Flyway - state: escape
        (0x98F3, 0x09, 0x05), // room $98E2 - Pre-Map Flyway
        (0x991E, 0x09, 0x05), // room $990D - Terminator Room
        (0x9949, 0x09, 0x03), // room $9938 - Green Brinstar Elevator Room
        (0x997A, 0x09, 0x05), // room $9969 - Lower Mushrooms
        (0x99A5, 0x09, 0x05), // room $9994 - Crateria Map Room
        (0x99CE, 0x09, 0x05), // room $99BD - Green Pirates Shaft
        (0x9A0A, 0x09, 0x05), // room $99F9 - Crateria Super Room
        (0x9A5A, 0x06, 0x05), // room $9A44 - Final Missile Bombway (state: Zebes not awake)
        (0x9A74, 0x09, 0x05), // room $9A44 - Final Missile Bombway (state: Zebes awake)
        (0x9AA6, 0x06, 0x05), // room $9A90 - The Final Missile (state: Zebes not awake)
        (0x9AC0, 0x09, 0x05), // room $9A90 - The Final Missile (state: Zebes awake)
        // Brinstar
        (0x9AEA, 0x0F, 0x05), // room $9AD9 - Green Brinstar Main Shaft
        (0x9B6C, 0x00, 0x03), // room $9B5B - Spore Spawn Super Room
        (0x9BAE, 0x0F, 0x05), // room $9B9D - Brinstar Pre-Map Room
        (0x9BD9, 0x0F, 0x05), // room $9BC8 - Early Supers Room
        (0x9C18, 0x00, 0x03), // room $9C07 - Brinstar Reserve Tank Room
        (0x9C46, 0x0F, 0x05), // room $9C35 - Brinstar Map Room
        (0x9C6F, 0x0F, 0x05), // room $9C5E - Green Brinstar Fireflea Room
        (0x9C9A, 0x0F, 0x05), // room $9C89 - Green Brinstar Missile Refill Room
        (0x9CC4, 0x0F, 0x05), // room $9CB3 - Dachora Room
        (0x9D2A, 0x0F, 0x05), // room $9D19 - Big Pink
        (0x9DAD, 0x0F, 0x05), // room $9D9C - Spore Spawn Kihunter Room
        (0x9DDD, 0x2A, 0x05), // room $9DC7 - Spore Spawn Room - state: spore spawn not dead
        (0x9DF7, 0x00, 0x03), // room $9DC7 - Spore Spawn Room - state: spore spawn dead
        (0x9E22, 0x0F, 0x05), // room $9E11 - Pink Brinstar Power Bomb Room
        (0x9E63, 0x0F, 0x05), // room $9E52 - Green Hill Zone
        (0x9EB5, 0x06, 0x07), // room $9E9F - Morph Ball Room - state: Zebes not awake
        (0x9ECF, 0x09, 0x05), // room $9E9F - Morph Ball Room - state: Zebes awake
        (0x9F27, 0x06, 0x07), // room $9F11 - Construction Zone (state: Zebes not awake)
        (0x9F41, 0x09, 0x05), // room $9F11 - Construction Zone (state: Zebes awake)
        (0x9F7A, 0x06, 0x07), // room $9F64 - Blue Brinstar Energy Tank Room (state: Zebes not awake)
        (0x9F94, 0x09, 0x05), // room $9F64 - Blue Brinstar Energy Tank Room (state: Zebes awake)
        (0x9FCB, 0x0F, 0x05), // room $9FBA - Noob Bridge
        (0x9FF6, 0x0F, 0x05), // room $9FE5 - Green Brinstar Beetom Room
        (0xA022, 0x0F, 0x05), // room $A011 - Etecoon Energy Tank Room
        (0xA062, 0x0F, 0x05), // room $A051 - Etecoon Super Room
        (0xA08C, 0x0F, 0x05), // room $A07B - Dachora Energy Refill Room
        (0xA0B5, 0x0F, 0x05), // room $A0A4 - Spore Spawn Farming Room
        (0xA0E3, 0x0F, 0x05), // room $A0D2 - Waterway Energy Tank Room
        (0xA118, 0x09, 0x05), // room $A107 - First Missile Room (using song from when Zebes awake)
        (0xA141, 0x0F, 0x05), // room $A130 - Pink Brinstar Hopper Room
        (0xA16C, 0x0F, 0x05), // room $A15B - Hopper Energy Tank Room
        (0xA195, 0x0F, 0x05), // room $A184 - Big Pink Save Room
        (0xA1BE, 0x09, 0x05), // room $A1AD - Blue Brinstar Boulder Room
        (0xA1E9, 0x09, 0x05), // room $A1D8 - Billy Mays Room
        (0xA212, 0x0F, 0x05), // room $A201 - Green Brinstar Main Shaft Save Room
        (0xA23B, 0x0F, 0x05), // room $A22A - Etecoon Save Room
        (0xA264, 0x12, 0x05), // room $A253 - Red Tower
        (0xA2A4, 0x12, 0x05), // room $A293 - Red Brinstar Fireflea Room
        (0xA2DF, 0x00, 0x03), // room $A2CE - X-Ray Scope Room
        (0xA308, 0x12, 0x05), // room $A2F7 - Hellway
        (0xA333, 0x12, 0x05), // room $A322 - Caterpillar Room
        (0xA38D, 0x12, 0x05), // room $A37C - Beta Power Bomb Room
        (0xA3BF, 0x00, 0x03), // room $A3AE - Alpha Power Bomb Room
        (0xA3EE, 0x12, 0x05), // room $A3DD - Bat Room
        (0xA419, 0x12, 0x05), // room $A408 - Below Spazer
        (0xA458, 0x00, 0x03), // room $A447 - Spazer Room
        (0xA482, 0x12, 0x05), // room $A471 - Warehouse Zeela Room
        (0xA4C2, 0x12, 0x05), // room $A4B1 - Warehouse Energy Tank Room
        (0xA4EB, 0x12, 0x05), // room $A4DA - Warehouse Kihunter Room
        (0xA537, 0x27, 0x06), // room $A521 - Baby Kraid Room - state: kraid not dead
        (0xA551, 0x27, 0x03), // room $A521 - Baby Kraid Room - state: kraid dead
        (0xA57C, 0x12, 0x05), // room $A56B - Kraid Eye Door Room (using generic Red Brinstar song)
        (0xA5B5, 0x27, 0x06), // room $A59F - Kraid Room - state: kraid not dead
        (0xA5CF, 0x00, 0x03), // room $A59F - Kraid Room - state: kraid dead
        (0xA5FE, 0x00, 0x04), // room $A5ED - Statues Hallway
        (0xA629, 0x12, 0x05), // room $A618 - Sloaters Refill
        (0xA652, 0x12, 0x05), // room $A641 - Kraid Recharge Station
        (0xA67B, 0x09, 0x06), // room $A66A - Statues Room
        (0xA6B2, 0x12, 0x03), // room $A6A1 - Warehouse Entrance
        (0xA6F3, 0x00, 0x03), // room $A6E2 - Varia Suit Room
        (0xA71C, 0x12, 0x05), // room $A70B - Warehouse Save Room
        (0xA745, 0x12, 0x05), // room $A734 - Caterpillar Save Room
        // Upper Norfair
        (0xA76E, 0x15, 0x05), // room $A75D - Ice Beam Acid Room
        (0xA799, 0x15, 0x05), // room $A788 - Cathedral
        (0xA7C4, 0x15, 0x05), // room $A7B3 - Cathedral Entrance
        (0xA826, 0x15, 0x05), // room $A815 - Ice Beam Gate Room
        (0xA876, 0x15, 0x05), // room $A865 - Ice Beam Tutorial Room
        (0xA8CA, 0x15, 0x05), // room $A8B9 - Ice Beam Snake Room
        (0xA909, 0x15, 0x05), // room $A8F8 - Crumble Shaft
        (0xAA1F, 0x15, 0x05), // room $AA0E - Crocomire Escape
        (0xAA52, 0x15, 0x05), // room $AA41 - Hi Jump Energy Tank Room
        (0xAAEF, 0x15, 0x05), // room $AADE - Post Crocomire Power Bomb Room
        (0xAB18, 0x15, 0x05), // room $AB07 - Post Crocomire Shaft
        (0xAB4C, 0x15, 0x05), // room $AB3B - Post Crocomire Missile Room
        (0xAB75, 0x15, 0x05), // room $AB64 - Grapple Tutorial Room 3
        (0xABA0, 0x15, 0x05), // room $AB8F - Post Crocomire Jump Room
        (0xABE3, 0x15, 0x05), // room $ABD2 - Grapple Tutorial Room 2
        (0xAC11, 0x15, 0x05), // room $AC00 - Grapple Tutorial Room 1
        (0xAC6B, 0x15, 0x05), // room $AC5A - Norfair Reserve Tank Room
        (0xAC94, 0x15, 0x05), // room $AC83 - Green Bubbles Missile Room
        (0xACC4, 0x15, 0x05), // room $ACB3 - Bubble Mountain
        (0xAD01, 0x15, 0x05), // room $ACF0 - Speed Booster Hall
        (0xADBE, 0x15, 0x05), // room $ADAD - Double Chamber
        (0xAE18, 0x15, 0x05), // room $AE07 - Spiky Platforms Tunnel
        (0xAE43, 0x15, 0x05), // room $AE32 - Volcano Room
        (0xAE85, 0x15, 0x05), // room $AE74 - Kronic Boost Room
        (0xAEC5, 0x15, 0x05), // room $AEB4 - Magdollite Tunnel
        (0xAEF0, 0x15, 0x05), // room $AEDF - Purple Shaft
        (0xAF25, 0x15, 0x05), // room $AF14 - Lava Dive Room
        (0xAF83, 0x15, 0x05), // room $AF72 - Upper Norfair Farming Room
        (0xAFB4, 0x15, 0x05), // room $AFA3 - Rising Tide
        (0xAFDF, 0x15, 0x05), // room $AFCE - Acid Snakes Tunnel
        (0xB00C, 0x15, 0x05), // room $AFFB - Spiky Acid Snakes Tunnel
        (0xB037, 0x15, 0x05), // room $B026 - Nutella Refill
        (0xB062, 0x15, 0x05), // room $B051 - Purple Farming Room
        (0xB08B, 0x15, 0x05), // room $B07A - Bat Cave
        (0xB0C5, 0x15, 0x05), // room $B0B4 - Norfair Map Room
        (0xB117, 0x15, 0x05), // room $B106 - Frog Speedway
        (0xB14A, 0x15, 0x05), // room $B139 - Red Pirate Shaft
        // Lower Norfair
        (0xB3B6, 0x18, 0x05), // room $B3A5 - Fast Pillars Setup Room
        (0xB41B, 0x18, 0x05), // room $B40A - Mickey Mouse Room
        (0xB468, 0x18, 0x05), // room $B457 - Pillar Room
        (0xB493, 0x18, 0x05), // room $B482 - Plowerhouse Room
        (0xB4BE, 0x18, 0x05), // room $B4AD - The Worst Room In The Game
        (0xB4F6, 0x18, 0x05), // room $B4E5 - Amphitheatre
        (0xB521, 0x18, 0x05), // room $B510 - Lower Norfair Spring Ball Maze Room
        (0xB56B, 0x18, 0x05), // room $B55A - Lower Norfair Escape Power Bomb Room
        (0xB596, 0x18, 0x05), // room $B585 - Red Kihunter Shaft
        (0xB5E6, 0x18, 0x05), // room $B5D5 - Wasteland
        (0xB63C, 0x18, 0x05), // room $B62B - Metal Pirates Room
        (0xB6A9, 0x00, 0x03), // room $B698 - Ridley Tank Room
        // Wrecked Ship
        (0xCAC4, 0x30, 0x05), // room $CAAE - Assembly Line (power off)
        (0xCADE, 0x30, 0x06), // room $CAAE - Assembly Line (power on)
        (0xCBA1, 0x30, 0x05), // room $CB8B - Spiky Death Room (power off)
        (0xCBBB, 0x30, 0x06), // room $CB8B - Spiky Death Room (power on)
        (0xCCE1, 0x30, 0x05), // room $CCCB - Wrecked Ship Map Room (power off)
        (0xCCFB, 0x30, 0x06), // room $CCCB - Wrecked Ship Map Room (power on)
        (0xCD72, 0x30, 0x05), // room $CD5C - Sponge Bath (power off)
        (0xCD8C, 0x30, 0x06), // room $CD5C - Sponge Bath (power on)
        (0xCDBE, 0x30, 0x05), // room $CDA8 - Wrecked Ship West Super Room (power off)
        (0xCDD8, 0x30, 0x06), // room $CDA8 - Wrecked Ship West Super Room (power on)
        (0xCE07, 0x30, 0x05), // room $CDF1 - Wrecked Ship East Super Room (power off)
        (0xCE21, 0x30, 0x06), // room $CDF1 - Wrecked Ship East Super Room (power on)
        (0xCE56, 0x30, 0x05), // room $CE40 - Gravity Suit Room (power off)
        (0xCE70, 0x30, 0x06), // room $CE40 - Gravity Suit Room (power on)
        // Maridia
        (0xCF65, 0x12, 0x05), // room $CF54 - West Tunnel
        (0xD028, 0x1B, 0x06), // room $D017 - Fish Tank
        (0xD066, 0x1B, 0x06), // room $D055 - Mama Turtle Room
        (0xD0CA, 0x1B, 0x06), // room $D0B9 - Mt. Everest
        (0xD14C, 0x1B, 0x06), // room $D13B - Watering Hole
        (0xD17E, 0x1B, 0x06), // room $D16D - Northwest Maridia Bug Room
        (0xD1EE, 0x1B, 0x06), // room $D1DD - Pseudo Plasma Spark Room
        (0xD263, 0x1B, 0x05), // room $D252 - West Sand Hall Tunnel
        (0xD28F, 0x1B, 0x05), // room $D27E - Plasma Tutorial Room
        (0xD2BB, 0x1B, 0x05), // room $D2AA - Plasma Room
        (0xD2EA, 0x1B, 0x05), // room $D2D9 - Thread The Needle Room
        (0xD351, 0x1B, 0x05), // room $D340 - Plasma Spark Room
        (0xD398, 0x1B, 0x05), // room $D387 - Kassiuz Room
        (0xD3C7, 0x1B, 0x05), // room $D3B6 - Maridia Map Room
        (0xD419, 0x1B, 0x05), // room $D408 - Toilet
        (0xD444, 0x1B, 0x05), // room $D433 - Bug Sand Hole
        (0xD472, 0x1B, 0x05), // room $D461 - West Sand Hall
        (0xD49F, 0x1B, 0x05), // room $D48E - Oasis
        (0xD4D3, 0x1B, 0x05), // room $D4C2 - East Sand Hall
        (0xD500, 0x1B, 0x05), // room $D4EF - West Sand Hole
        (0xD52F, 0x1B, 0x05), // room $D51E - East Sand Hole
        (0xD55E, 0x1B, 0x05), // room $D54D - West Aqueduct Quicksand Room
        (0xD58B, 0x1B, 0x05), // room $D57A - East Aqueduct Quicksand Room
        (0xD5FD, 0x1B, 0x05), // room $D5EC - Butterfly Room
        (0xD657, 0x1B, 0x05), // room $D646 - Pants Room
        (0xD6AB, 0x1B, 0x05), // room $D69A - East Pants Room
        (0xD6E1, 0x1B, 0x05), // room $D6D0 - Spring Ball Room
        (0xD70E, 0x1B, 0x05), // room $D6FD - Below Botwoon Energy Tank
        (0xD73B, 0x1B, 0x05), // room $D72A - Colosseum
        (0xD7A5, 0x1B, 0x05), // room $D78F - The Precious Room
        (0xD856, 0x1B, 0x05), // room $D845 - Maridia Missile Refill Room
        (0xD87F, 0x1B, 0x05), // room $D86E - Plasma Beach Quicksand Room
        (0xD8A9, 0x1B, 0x05), // room $D898 - Botwoon Quicksand Room
        (0xD8DB, 0x1B, 0x05), // room $D8C5 - Shaktool Room
        (0xD8F5, 0x1B, 0x05), // room $D8C5 - Shaktool Room (state: sand gone)
        (0xD924, 0x1B, 0x05), // room $D913 - Halfie Climb Room
        (0xD9E5, 0x1B, 0x05), // room $D9D4 - Maridia Health Refill Room
        (0xDA0F, 0x1B, 0x05), // room $D9FE - West Cactus Alley Room
        (0xDA3C, 0x1B, 0x05), // room $DA2B - East Cactus Alley Room
        // Tourian
        (0xDAF7, 0x1E, 0x05), // room $DAE1 - Metroid Room 1
        (0xDB11, 0x1E, 0x05), // room $DAE1 - Metroid Room 1 (state: metroids gone)
        (0xDB47, 0x1E, 0x05), // room $DB31 - Metroid Room 2
        (0xDB61, 0x1E, 0x05), // room $DB31 - Metroid Room 2 (state: metroids gone)
        (0xDB93, 0x1E, 0x05), // room $DB7D - Metroid Room 3
        (0xDBAD, 0x1E, 0x05), // room $DB7D - Metroid Room 3 (state: metroids gone)
        (0xDCC7, 0x45, 0x06), // room $DCB1 - Big Boy Room (state: normal state)
        (0xDCE1, 0x45, 0x06), // room $DCB1 - Big Boy Room (state: unused game state. We update this anyway.)
        (0xDD3F, 0x1E, 0x05), // room $DD2E - Tourian Recharge Room
        (0xDD72, 0x1E, 0x05), // room $DD58 - Mother Brain Room
        (0xDD8C, 0x1E, 0x05), // room $DD58 - Mother Brain Room (state: glass broken)
        (0xDDD5, 0x1E, 0x05), // room $DDC4 - Tourian Eye Door Room
        (0xDE5E, 0x1E, 0x05), // room $DE4D - Tourian Escape Room 1
        (0xDE8B, 0x1E, 0x05), // room $DE7A - Tourian Escape Room 2
        (0xDEB8, 0x1E, 0x05), // room $DE7A - Tourian Escape Room 3
        (0xDEEF, 0x1E, 0x05), // room $DE7A - Tourian Escape Room 4  
    ];
    Ok(())
}
