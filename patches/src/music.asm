; In the vanilla game, many rooms have their songset and/or play index as "no change", so they depend on surrounding
; rooms to start the correct song. This would give unexpected results when we rearrange the rooms, so we replace
; "no change" with a specific songset & play-index (usually the one from the vanilla game).
;
; Crateria
org $8F92C9 : db $06, $05  ; room $92B3 - Gauntlet Entrance (state: Zebes not awake)
org $8F9318 : db $06, $05  ; room $92FD - Parlor and Alcatraz (state: Zebes not awake)
org $8F93BB : db $09, $05  ; room $93AA - Crateria Power Bomb Room
org $8F9472 : db $0C, $05  ; room $9461 - Bowling Alley Path
org $8F949D : db $09, $05  ; room $948C - Crateria Kihunter Room
org $8F9563 : db $0C, $05  ; room $9552 - Forgotten Highway Kago Room
org $8F958E : db $0C, $05  ; room $957D - Crab Maze
org $8F95B9 : db $0C, $05  ; room $95A8 - Forgotten Highway Elbow
org $8F95E5 : db $09, $05  ; room $95D4 - Crateria Tube
org $8F9610 : db $09, $05  ; room $95FF - The Moat
org $8F966C : db $09, $05  ; room $965B - Gauntlet Energy Tank Room
org $8F96D5 : db $06, $05  ; room $96BA - Climb (state: Zebes not awake)
org $8F9771 : db $06, $05  ; room $975C - Pit Room (state: Zebes not awake)
org $8F98F3 : db $09, $05  ; room $98E2 - Pre-Map Flyway
org $8F991E : db $09, $05  ; room $990D - Terminator Room
org $8F997A : db $09, $05  ; room $9969 - Lower Mushrooms
org $8F99A5 : db $09, $05  ; room $9994 - Crateria Map Room
org $8F99CE : db $09, $05  ; room $99BD - Green Pirates Shaft
org $8F9A0A : db $09, $05  ; room $99F9 - Crateria Super Room
org $8F9A5A : db $06, $05  ; room $9A44 - Final Missile Bombway (state: Zebes not awake)
org $8F9A74 : db $09, $05  ; room $9A44 - Final Missile Bombway (state: Zebes awake)
org $8F9AA6 : db $06, $05  ; room $9A90 - The Final Missile (state: Zebes not awake)
org $8F9AC0 : db $09, $05  ; room $9A90 - The Final Missile (state: Zebes awake)
; Brinstar
org $8F9BAE : db $0F, $05  ; room $9B9D - Brinstar Pre-Map Room
org $8F9BD9 : db $0F, $05  ; room $9BC8 - Early Supers Room
org $8F9C46 : db $0F, $05  ; room $9C35 - Brinstar Map Room
org $8F9C6F : db $0F, $05  ; room $9C5E - Green Brinstar Fireflea Room
org $8F9C9A : db $0F, $05  ; room $9C89 - Green Brinstar Missile Refill Room
org $8F9CC4 : db $0F, $05  ; room $9CB3 - Dachora Room
org $8F9E22 : db $0F, $05  ; room $9E11 - Pink Brinstar Power Bomb Room
org $8F9F27 : db $06, $07  ; room $9F11 - Construction Zone (state: Zebes not awake)
org $8F9F41 : db $09, $05  ; room $9F11 - Construction Zone (state: Zebes awake)
org $8F9F7A : db $06, $07  ; room $9F64 - Blue Brinstar Energy Tank Room (state: Zebes not awake)
org $8F9F94 : db $09, $05  ; room $9F64 - Blue Brinstar Energy Tank Room (state: Zebes awake)
org $8F9FF6 : db $0F, $05  ; room $9FE5 - Green Brinstar Beetom Room
org $8FA022 : db $0F, $05  ; room $A011 - Etecoon Energy Tank Room
org $8FA062 : db $0F, $05  ; room $A051 - Etecoon Super Room
org $8FA08C : db $0F, $05  ; room $A07B - Dachora Energy Refill Room
org $8FA0E3 : db $0F, $05  ; room $A0D2 - Waterway Energy Tank Room
org $8FA118 : db $09, $05  ; room $A107 - First Missile Room (using song from when Zebes awake)
org $8FA141 : db $0F, $05  ; room $A130 - Pink Brinstar Hopper Room
org $8FA16C : db $0F, $05  ; room $A15B - Hopper Energy Tank Room
org $8FA1BE : db $09, $05  ; room $A1AD - Blue Brinstar Boulder Room
org $8FA1E9 : db $09, $05  ; room $A1D8 - Billy Mays Room
org $8FA2A4 : db $12, $05  ; room $A293 - Red Brinstar Fireflea Room
org $8FA308 : db $12, $05  ; room $A2F7 - Hellway
org $8FA38D : db $12, $05  ; room $A37C - Beta Power Bomb Room
org $8FA3EE : db $12, $05  ; room $A3DD - Bat Room
org $8FA419 : db $12, $05  ; room $A408 - Below Spazer
org $8FA482 : db $12, $05  ; room $A471 - Warehouse Zeela Room
org $8FA4C2 : db $12, $05  ; room $A4B1 - Warehouse Energy Tank Room
org $8FA57C : db $12, $05  ; room $A56B - Kraid Eye Door Room (using generic Red Brinstar song)
org $8FA5CF : db $00, $03  ; room $A59F - Kraid Room (boss dead)
org $8FA629 : db $12, $05  ; room $A618 - Sloaters Refill
org $8FA652 : db $12, $05  ; room $A641 - Kraid Recharge Station
; Upper Norfair
org $8FA76E : db $15, $05  ; room $A75D - Ice Beam Acid Room
org $8FA799 : db $15, $05  ; room $A788 - Cathedral
org $8FA7C4 : db $15, $05  ; room $A7B3 - Cathedral Entrance
org $8FA826 : db $15, $05  ; room $A815 - Ice Beam Gate Room
org $8FA876 : db $15, $05  ; room $A865 - Ice Beam Tutorial Room
org $8FA8CA : db $15, $05  ; room $A8B9 - Ice Beam Snake Room
org $8FA909 : db $15, $05  ; room $A8F8 - Crumble Shaft
org $8FAA1F : db $15, $05  ; room $AA0E - Crocomire Escape
org $8FAA52 : db $15, $05  ; room $AA41 - Hi Jump Energy Tank Room
org $8FAAEF : db $15, $05  ; room $AADE - Post Crocomire Power Bomb Room
org $8FAB18 : db $15, $05  ; room $AB07 - Post Crocomire Shaft
org $8FAB4C : db $15, $05  ; room $AB3B - Post Crocomire Missile Room
org $8FAB75 : db $15, $05  ; room $AB64 - Grapple Tutorial Room 3
org $8FABA0 : db $15, $05  ; room $AB8F - Post Crocomire Jump Room
org $8FABE3 : db $15, $05  ; room $ABD2 - Grapple Tutorial Room 2
org $8FAC11 : db $15, $05  ; room $AC00 - Grapple Tutorial Room 1
org $8FAC6B : db $15, $05  ; room $AC5A - Norfair Reserve Tank Room
org $8FAC94 : db $15, $05  ; room $AC83 - Green Bubbles Missile Room
org $8FACC4 : db $15, $05  ; room $ACB3 - Bubble Mountain
org $8FAD01 : db $15, $05  ; room $ACF0 - Speed Booster Hall
org $8FADBE : db $15, $05  ; room $ADAD - Double Chamber
org $8FAE18 : db $15, $05  ; room $AE07 - Spiky Platforms Tunnel
org $8FAE43 : db $15, $05  ; room $AE32 - Volcano Room
org $8FAE85 : db $15, $05  ; room $AE74 - Kronic Boost Room
org $8FAEC5 : db $15, $05  ; room $AEB4 - Magdollite Tunnel
org $8FAEF0 : db $15, $05  ; room $AEDF - Purple Shaft
org $8FAF25 : db $15, $05  ; room $AF14 - Lava Dive Room
org $8FAF83 : db $15, $05  ; room $AF72 - Upper Norfair Farming Room
org $8FAFB4 : db $15, $05  ; room $AFA3 - Rising Tide
org $8FAFDF : db $15, $05  ; room $AFCE - Acid Snakes Tunnel
org $8FB00C : db $15, $05  ; room $AFFB - Spiky Acid Snakes Tunnel
org $8FB037 : db $15, $05  ; room $B026 - Nutella Refill
org $8FB062 : db $15, $05  ; room $B051 - Purple Farming Room
org $8FB08B : db $15, $05  ; room $B07A - Bat Cave
org $8FB0C5 : db $15, $05  ; room $B0B4 - Norfair Map Room
org $8FB117 : db $15, $05  ; room $B106 - Frog Speedway
org $8FB14A : db $15, $05  ; room $B139 - Red Pirate Shaft
; Lower Norfair
org $8FB3B6 : db $18, $05  ; room $B3A5 - Fast Pillars Setup Room
org $8FB41B : db $18, $05  ; room $B40A - Mickey Mouse Room
org $8FB468 : db $18, $05  ; room $B457 - Pillar Room
org $8FB493 : db $18, $05  ; room $B482 - Plowerhouse Room
org $8FB4BE : db $18, $05  ; room $B4AD - The Worst Room In The Game
org $8FB4F6 : db $18, $05  ; room $B4E5 - Amphitheatre
org $8FB521 : db $18, $05  ; room $B510 - Lower Norfair Spring Ball Maze Room
org $8FB56B : db $18, $05  ; room $B55A - Lower Norfair Escape Power Bomb Room
org $8FB596 : db $18, $05  ; room $B585 - Red Kihunter Shaft
org $8FB5E6 : db $18, $05  ; room $B5D5 - Wasteland
org $8FB63C : db $18, $05  ; room $B62B - Metal Pirates Room
org $8FB6A9 : db $00, $03  ; room $B698 - Ridley Tank Room
; Wrecked Ship
org $8FCAC4 : db $30, $05  ; room $CAAE - Assembly Line (power off)
org $8FCADE : db $30, $06  ; room $CAAE - Assembly Line (power on)
org $8FCBA1 : db $30, $05  ; room $CB8B - Spiky Death Room (power off)
org $8FCBBB : db $30, $06  ; room $CB8B - Spiky Death Room (power on)
org $8FCCE1 : db $30, $05  ; room $CCCB - Wrecked Ship Map Room (power off)
org $8FCCFB : db $30, $06  ; room $CCCB - Wrecked Ship Map Room (power on)
org $8FCD72 : db $30, $05  ; room $CD5C - Sponge Bath (power off)
org $8FCD8C : db $30, $06  ; room $CD5C - Sponge Bath (power on)
org $8FCDBE : db $30, $05  ; room $CDA8 - Wrecked Ship West Super Room (power off)
org $8FCDD8 : db $30, $06  ; room $CDA8 - Wrecked Ship West Super Room (power on)
org $8FCE07 : db $30, $05  ; room $CDF1 - Wrecked Ship East Super Room (power off)
org $8FCE21 : db $30, $06  ; room $CDF1 - Wrecked Ship East Super Room (power on)
org $8FCE56 : db $30, $05  ; room $CE40 - Gravity Suit Room (power off)
org $8FCE70 : db $30, $06  ; room $CE40 - Gravity Suit Room (power on)
; Maridia
org $8FCF65 : db $12, $05  ; room $CF54 - West Tunnel
org $8FD028 : db $1B, $06  ; room $D017 - Fish Tank
org $8FD066 : db $1B, $06  ; room $D055 - Mama Turtle Room
org $8FD0CA : db $1B, $06  ; room $D0B9 - Mt. Everest
org $8FD14C : db $1B, $06  ; room $D13B - Watering Hole
org $8FD17E : db $1B, $06  ; room $D16D - Northwest Maridia Bug Room
org $8FD1EE : db $1B, $06  ; room $D1DD - Pseudo Plasma Spark Room
org $8FD263 : db $1B, $05  ; room $D252 - West Sand Hall Tunnel
org $8FD28F : db $1B, $05  ; room $D27E - Plasma Tutorial Room
org $8FD2BB : db $1B, $05  ; room $D2AA - Plasma Room
org $8FD2EA : db $1B, $05  ; room $D2D9 - Thread The Needle Room
org $8FD351 : db $1B, $05  ; room $D340 - Plasma Spark Room
org $8FD398 : db $1B, $05  ; room $D387 - Kassiuz Room
org $8FD3C7 : db $1B, $05  ; room $D3B6 - Maridia Map Room
org $8FD419 : db $1B, $05  ; room $D408 - Toilet
org $8FD444 : db $1B, $05  ; room $D433 - Bug Sand Hole
org $8FD472 : db $1B, $05  ; room $D461 - West Sand Hall
org $8FD49F : db $1B, $05  ; room $D48E - Oasis
org $8FD4D3 : db $1B, $05  ; room $D4C2 - East Sand Hall
org $8FD500 : db $1B, $05  ; room $D4EF - West Sand Hole
org $8FD52F : db $1B, $05  ; room $D51E - East Sand Hole
org $8FD55E : db $1B, $05  ; room $D54D - West Aqueduct Quicksand Room
org $8FD58B : db $1B, $05  ; room $D57A - East Aqueduct Quicksand Room
org $8FD5FD : db $1B, $05  ; room $D5EC - Butterfly Room
org $8FD657 : db $1B, $05  ; room $D646 - Pants Room
org $8FD6AB : db $1B, $05  ; room $D69A - East Pants Room
org $8FD6E1 : db $1B, $05  ; room $D6D0 - Spring Ball Room
org $8FD70E : db $1B, $05  ; room $D6FD - Below Botwoon Energy Tank
org $8FD73B : db $1B, $05  ; room $D72A - Colosseum
org $8FD7A5 : db $1B, $05  ; room $D78F - The Precious Room
org $8FD856 : db $1B, $05  ; room $D845 - Maridia Missile Refill Room
org $8FD87F : db $1B, $05  ; room $D86E - Plasma Beach Quicksand Room
org $8FD8A9 : db $1B, $05  ; room $D898 - Botwoon Quicksand Room
org $8FD8DB : db $1B, $05  ; room $D8C5 - Shaktool Room
org $8FD8F5 : db $1B, $05  ; room $D8C5 - Shaktool Room (state: sand gone)
org $8FD924 : db $1B, $05  ; room $D913 - Halfie Climb Room
org $8FD9E5 : db $1B, $05  ; room $D9D4 - Maridia Health Refill Room
org $8FDA0F : db $1B, $05  ; room $D9FE - West Cactus Alley Room
org $8FDA3C : db $1B, $05  ; room $DA2B - East Cactus Alley Room
; Tourian
org $8FDAF7 : db $1E, $05  ; room $DAE1 - Metroid Room 1
org $8FDB11 : db $1E, $05  ; room $DAE1 - Metroid Room 1 (state: metroids gone)
org $8FDB47 : db $1E, $05  ; room $DB31 - Metroid Room 2
org $8FDB61 : db $1E, $05  ; room $DB31 - Metroid Room 2 (state: metroids gone)
org $8FDB93 : db $1E, $05  ; room $DB7D - Metroid Room 3
org $8FDBAD : db $1E, $05  ; room $DB7D - Metroid Room 3 (state: metroids gone)
org $8FDCC7 : db $45, $06  ; room $DCB1 - Big Boy Room (state: pre-cutscene)
org $8FDCE1 : db $1E, $05  ; room $DCB1 - Big Boy Room (state: post-cutscene)
org $8FDD3F : db $1E, $05  ; room $DD2E - Tourian Recharge Room
org $8FDD72 : db $1E, $05  ; room $DD58 - Mother Brain Room
org $8FDD8C : db $1E, $05  ; room $DD58 - Mother Brain Room (state: glass broken)
org $8FDDD5 : db $1E, $05  ; room $DDC4 - Tourian Eye Door Room
org $8FDE5E : db $1E, $05  ; room $DE4D - Tourian Escape Room 1
org $8FDE8B : db $1E, $05  ; room $DE7A - Tourian Escape Room 2
org $8FDEB8 : db $1E, $05  ; room $DE7A - Tourian Escape Room 3
org $8FDEEF : db $1E, $05  ; room $DE7A - Tourian Escape Room 4
