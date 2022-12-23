arch snes.cpu
lorom

org $83AAEA
    dw $EE00               ; Set door ASM for Tourian Escape Room 1 toward Mother Brain

org $8FEE00
    JSL $8483D7            ;\
    db  $00, $06           ;|
    dw  $B677              ;} Spawn Mother Brain's room escape door
