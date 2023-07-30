arch snes.cpu
lorom

; Must match locations in patch/map_tiles.rs
!item_list_ptrs = $83B000
!item_list_sizes = $83B00C

!bank_82_freespace_start = $82FD00
!bank_82_freespace_end = $82FD80

;org $8293E2
;    jsr load_pause_map_hook

org $82945C
    jsr transfer_pause_tilemap_hook

; patch HUD minimap drawing to copy from $703000 instead of original ROM tilemap, to pick up item dot modifications
org $90AA73
patch_hud_minimap_draw:
    lda #$0070
    sta $05
    sta $02
    sta $08
    lda #$3000
    bra .setup
warnpc $90AA8A
org $90AA8A
.setup:

; update tilemap when loading game:
org $80A0D9
    jsl update_tilemap  ; no hi-jack needed since the replaced instruction does nothing

org $829370
    jsl unpause_hook

org $82DFC2
    jsl reload_map_hook

org $85846D
    jsl message_box_hook   ; reload map after message boxes (e.g. after item collect or map station activation)
    nop

; Free space in bank $82:
org !bank_82_freespace_start

transfer_pause_tilemap_hook:
    jsl update_tilemap  ; update tilemap with collected item dots

    ; Set source tilemap to $703000 for further processing
    lda #$3000
    sta $00
    lda #$0070
    sta $02

    lda #$4000  ; run hi-jacked instruction
    rts

warnpc !bank_82_freespace_end

; Free space in any bank:
org $83B300

unpause_hook:
    jsl update_tilemap
    jsl $80A149  ; run hi-jacked instruction
    rtl

message_box_hook:
    jsl update_tilemap
    
    ; run hi-jacked instructions:
    sep #$20
    lda $1C1F
    rtl

reload_map_hook:
    jsl $80858C  ; run hi-jacked instruction (load map explored bits)
    jsl update_tilemap
    
    ; clear HUD minimap
    LDX #$0000             ;|
    lda #$381f
.clear_minimap_loop:
    STA $7EC63C,x          ;|
    STA $7EC67C,x          ;} HUD tilemap (1Ah..1Eh, 1..3) = 2C1Fh
    STA $7EC6BC,x          ;|
    INX                    ;|
    INX                    ;|
    CPX #$000A             ;|
    BMI .clear_minimap_loop

    ; update VRAM for HUD
    LDX $0330       ;\
    LDA #$00C0      ;|
    STA $D0,x       ;|
    INX             ;|
    INX             ;|
    LDA #$C608      ;|
    STA $D0,x       ;|
    INX             ;|
    INX             ;} Queue transfer of $7E:C608..C7 to VRAM $5820..7F (HUD tilemap)
    LDA #$007E      ;|
    STA $D0,x       ;|
    INX             ;|
    LDA #$5820      ;|
    STA $D0,x       ;|
    INX             ;|
    INX             ;|
    STX $0330       ;/

    rtl

update_tilemap:
    php
    rep #$30

    ; $00 <- long pointer to original map tilemap in ROM, for current area
    lda $1F5B  ; current map area
    asl A
    clc
    adc $1F5B
    tax
    lda $82964C,x
    sta $02
    lda $82964A,x
    sta $00

    ; Copy tilemap from [$00] (original map tilemap in ROM) to $703000 (in SRAM, which we're just using as extra RAM)
    ldx #$0800  ; loop counter
    ldy #$0000  ; offset

    lda #$3000
    sta $03
    lda #$0070
    sta $05

.copy_loop:
    lda [$00], y
    sta [$03], y
    iny
    iny
    dex
    bne .copy_loop

    lda $1F5B
    asl
    tax
    lda !item_list_sizes, x
    beq .done  ; If there are no items in this area, then we're done.
    sta $06    ; $06 <- loop counter
    lda !item_list_ptrs, x
    tax  ; X <- item data offset

.item_loop:
    lda #$0000
    sep #$20
    lda $830000, x
    phx
    tax
    lda $7ED870, x
    plx
    and $830001, x
    bne .skip  ; item is collected, so skip overwriting tile

    rep #$20
    lda $830002, x  ; A <- tilemap offset
    tay
    lda $830004, x  ; A <- tilemap word
    sta [$03], y

.skip:
    rep #$20
    inx : inx : inx : inx : inx : inx
    dec $06
    bne .item_loop

.done:
    plp
    rtl