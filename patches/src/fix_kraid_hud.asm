; Kraid exit HUD corruption fix for SM Map Rando
; 
; Occurs due to Kraid room moving HUD (BG3) tileset address to $2000, then back to $4000 on room exit.
;
; -Stag Shot

arch 65816
lorom

!bank_82_free_space_start = $82fbb0
!bank_82_free_space_end = $82fd00

org $82de1d
    jmp room_ptr_hook

org $8883dc
    jml write_hud_vram

org !bank_82_free_space_start
room_ptr_hook:              ; A = next room; $79b = current room
    pha
    cmp #$a59f              ; entering kraid?
    beq .init_flag
    lda $79b
    cmp #$a59f              ; leaving kraid?
    beq .inc_flag
    bra .exit

.init_flag
    lda #$0001
    sta $009a
    bra .exit

.inc_flag
    inc $009a

.exit
    pla
    sta $079b               ; replaced code
    jmp $de20

write_hud_vram:
    pha
    phx
    php
    lda $009a
    cmp #$02                ; left kraid?
    bne .leave

    lda #$80
    sta $2100

    phb
    pea $8000 : plb : plb   ; set DB for reserve_hud
    rep #$30
    jsr $ff03               ; reserve_hud repaint (skip inc $998)
    plb

    stz $009a               ; clear temp flag

    sep #$20
    lda #$80
    sta $2115               ; 16-bit increment
    ldx #$4000
    stx $2116               ; VRAM dest
    ldx #$1801
    stx $4310
    ldx #$c000
    stx $4312               ; copy source (low)
    lda $1F5B               ; map area (0-5)
    clc
    adc #$E2
    sta $4314               ; copy source (high)
    ldx #$1000
    stx $4315               ; VRAM copy len
    lda #$02
    sta $420B               ; DMA 1 init

    lda #$0F
    sta $2100

.leave
    plp
    plx
    pla
    sta $5a                 ; replaced code
    sta $5b
    jml $8883e0

warnpc !bank_82_free_space_end
