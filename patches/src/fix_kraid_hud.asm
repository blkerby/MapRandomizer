; Kraid exit HUD corruption fix for SM Map Rando
; 
; Occurs due to Kraid room moving HUD (BG3) tileset address to $2000, then back to $4000 on room exit.
; Since the doors are left open initially, this can create a graphical race condition leading to 
; transient corruption. We fix this by doing a DMA write earlier in the door transition process.
;
; -Stag Shot

arch 65816
lorom

!bank_82_free_space_start = $82fbb0
!bank_82_free_space_end = $82fd00

org $82de1d
    jmp room_ptr_hook

org $8883d8
    jsl write_hud_vram

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

    stz $009a               ; clear temp flag

    lda #$80
    sta $2100

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
    sta $59                 ; replaced code
    lda #$5A
    rtl

warnpc !bank_82_free_space_end
