; In the vanilla game, some rooms have their songset and/or play index as "no change", so they depend on surrounding
; rooms to start the correct song. This would give unexpected results when we rearrange the rooms, so we replace
; "no change" with the specific songset & play-index that these rooms have in the vanilla game.
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

; Maridia
org $8FD14C : db $1B, $06  ; room $D13B - Watering Hole
