; In the vanilla game, some rooms have their songset and/or play index as "no change", so they depend on surrounding
; rooms to start the correct song. This would give unexpected results when we rearrange the rooms, so we replace
; "no change" with a specific songset & play-index (usually the one from in the vanilla game).
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
org $8F96D5 : db $09, $05  ; room $96BA - Climb (state: Zebes not awake)
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
org $8F9C96 : db $0F, $05  ; room $9C89 - Green Brinstar Missile Refill Room
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
org $8FA389 : db $12, $05  ; room $A37C - Beta Power Bomb Room
org $8FA3F2 : db $12, $05  ; room $A3DD - Bat Room
org $8FA415 : db $12, $05  ; room $A408 - Below Spazer
org $8FA482 : db $12, $05  ; room $A471 - Warehouse Zeela Room
org $8FA4C2 : db $12, $05  ; room $A4B1 - Warehouse Energy Tank Room
org $8FA57C : db $12, $05  ; room $A56B - Kraid Eye Door Room
org $8FA5CF : db $00, $03  ; room $A59F - Kraid Room (boss dead)
org $8FA629 : db $12, $05  ; room $A618 - Sloaters Refill
org $8FA652 : db $12, $05  ; room $A641 - Kraid Recharge Station

; Maridia
org $8FD14C : db $1B, $06  ; room $D13B - Watering Hole
