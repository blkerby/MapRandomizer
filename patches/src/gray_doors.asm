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
