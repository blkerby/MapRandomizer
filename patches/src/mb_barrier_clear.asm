arch snes.cpu
lorom

org $8FEB00
    ; setup asm to trigger gates in mother brain room based on main bosses killed
    lda #$0000
    ldx #$2000
loop:
    sta $7f0002,x
    inx
    cpx #$2100
    bne loop
    rts
