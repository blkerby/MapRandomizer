; Messing around here. This is broken.

arch snes.cpu
lorom


; hi-jack code that runs when selection YES on game-over screen.
org $81911E
    JMP skip_map_select

org $81F100
skip_map_select:
    LDA #$0006  ; Set game state = 6 (instead of 5)
    STA $0998
    LDA #$0000  ; menu index = 0
    STA $0727
    JML $81AD21