arch snes.cpu
lorom

org $8391C0  ; door ASM for entering Kraid Room from the left
    dw make_left_doors_blue

org $83925C  ; door ASM for entering Kraid Room from the right
    dw make_right_doors_blue

org $83A92E   ; door ASM for entering Draygon's Room from the left
    dw make_left_doors_blue

org $83A84A   ; door ASM for entering Draygon's Room from the right
    dw make_right_doors_blue

org $839A6C   ; door ASM for entering Ridley's Room from the left
    dw make_left_doors_blue

org $8398D4   ; door ASM for entering Ridley's Room from the right
    dw make_right_doors_blue

org $839A90   ; door ASM for entering Golden Torizo's Room from the right
    dw make_right_doors_blue

org $839184   ; door ASM for entering Baby Kraid Room from the left
    dw make_left_doors_blue

org $8391B4   ; door ASM for entering Baby Kraid Room from the right
    dw make_right_doors_blue

org $839970   ; door ASM for entering Metal Pirates from the left
    dw make_left_doors_blue

org $839A24   ; door ASM for entering Metal Pirates from the right
    dw make_right_doors_blue

; Replace Metal Pirates PLM set to add extra gray door on the right:
org $8FB64E
    dw metal_pirates_plms

org $8FF700
make_left_doors_blue:
    phx
    pha
    ldx #$0000
.loop
    lda $1C37, x
    cmp #$C848
    bne .next
    stz $1C37, x
.next
    inx : inx
    cpx #$0050
    bne .loop
    pla
    plx
    rts

make_right_doors_blue:
    phx
    pha
    ldx #$0000
.loop
    lda $1C37, x
    cmp #$C842
    bne .next
    stz $1C37, x
.next
    inx : inx
    cpx #$0050
    bne .loop
    pla
    plx
    rts

; Replaces PLM list at $8F90C8
metal_pirates_plms:
    ; left gray door:
    dw $C848
    db $01
    db $06
    dw $0C60
    ; right gray door:
    dw $C842
    db $1E
    db $06
    dw $0C60
    ; end marker:
    dw $0000

org $8FF800