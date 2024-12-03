arch snes.cpu
lorom

!bank_84_free_space_start = $84f580
!bank_84_free_space_end = $84f590

org $83AAEA
    dw $EE00               ; Set door ASM for Tourian Escape Room 1 toward Mother Brain

org $83AAE3
    db $00    ; Set door direction = $00  (to make door not close behind Samus)

; Custom door ASM for Tourian Escape Room 1 toward Mother Brain
org $8FEE00
    JSL $8483D7            ;\
    db  $00, $06           ;|
    dw  $B677              ;} Spawn Mother Brain's room escape door
    rts

; Restore escape door after MB1 in case of left entry
org $ADE3D1                ; hook MB1 => MB2 transition
    jsl fix_mb_escape_door

; instruction set to restore door
org $8494C9
    dw $8004, $8b0f, $8ae8, $82e8, $830f
    dw $0001
    dw $8004, $8b0f, $8ae8, $82e8, $830f
    dw $0000

org $84AC01
    dw #$94c9              ; switch to unused instruction set with adequate space

org $84B66F                ; custom PLM in unused space
    dw #$b3c1, #$abff      ; deactivate PLMs, reference above instruction set

org !bank_84_free_space_start
fix_mb_escape_door:
    pha

    jsl $8483d7            ; custom PLM that restores escape door
    db  $00, $06
    dw  $b66f

    pla
    sta $fa8               ; replaced code
    rtl

warnpc !bank_84_free_space_end
