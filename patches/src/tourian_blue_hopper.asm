arch snes.cpu
lorom

org $A3AB5D
    jsr adjust_hopper_spawn

org $A3F360
adjust_hopper_spawn:
    lda $079B
    cmp #$DC19
    bne .skip    ; skip if this isn't (Tourian) Blue Hopper Room
    lda $0791
    cmp #$0004
    bne .skip    ; skip if we aren't coming in the left door
    lda #$00C6   ; move enemy 2 X position (left hopper) over to the right
    sta $0FBA
.skip:
    lda #$0002   ; run hi-jacked instruction
    rts

warnpc $A3F3A0