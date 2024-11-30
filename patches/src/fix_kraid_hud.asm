; Kraid exit HUD corruption fix for SM Map Rando
; 
; Occurs due to Kraid room moving HUD (BG3) tileset address to $2000, then back to $4000 on room exit.
; Since the fight sequence is initially optional, exiting can create a graphical race condition leading
; to transient corruption. We fix this by doing a VRAM write earlier in the door transition process.
;
; -Stag Shot

arch 65816
lorom

!bank_82_free_space_start = $82fbb0
!bank_82_free_space_end = $82fd00

org $82de1d
    jsr room_ptr_hook

org !bank_82_free_space_start
room_ptr_hook:
    pha
    lda $79b                ; current room (leaving)
    sta $1f7e               ; save
    cmp #$a59f              ; leaving kraid?
    bne .exit

    phx
    php
    rep #$30
    ldx $0330
    lda #$1000                         : sta $00d0,x ; Number of bytes
    lda $1f5b : clc : adc #$00e2 : xba : sta $00d3,x ; Map area (high)
    lda #$c000                         : sta $00d2,x ; Source address (low)
    lda #$4000                         : sta $00d5,x ; Destination in Vram
    txa : clc : adc #$0007             : sta $0330   ; Update the stack pointer
    plp
    plx

.exit
    pla
    sta $079b               ; replaced code (new room)
    rts

warnpc !bank_82_free_space_end
