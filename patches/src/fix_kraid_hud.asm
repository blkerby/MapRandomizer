; Kraid exit HUD corruption fix for SM Map Rando
; 
; Occurs due to Kraid room moving HUD (BG3) tileset address to $2000, then back to $4000 on room exit.
; Since the fight sequence is initially optional, exiting can create a graphical race condition leading
; to transient corruption. We fix this by doing a VRAM write earlier in the door transition process.
;
; -Stag Shot

arch 65816
lorom

incsrc "constants.asm"

!bank_82_free_space_start = $82fbb0
!bank_82_free_space_end = $82fbf0

org $82de1d
    jsr room_ptr_hook

org !bank_82_free_space_start
room_ptr_hook:
    pha
    lda $79b                ; current room (leaving)
    sta !previous_room      ; save
    cmp #$a59f              ; leaving kraid?
    bne .exit

    phx
    php
    rep #$30
    ldx $0330
    lda #$0800                  : sta $00d0,x ; Number of bytes
    lda #$9a00                  : sta $00d3,x ; Source bank
    lda #$b200                  : sta $00d2,x ; Source address
    lda #$4000                  : sta $00d5,x ; Destination in Vram
    txa : clc : adc #$0007      : sta $0330   ; Update the stack pointer
    plp
    plx

    jsl $85A290             ; load_bg3_map_tiles_wrapper (map_area.asm)

.exit
    pla
    sta $079b               ; replaced code (new room)
    rts

warnpc !bank_82_free_space_end
