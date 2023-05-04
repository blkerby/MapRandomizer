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

; Free space in bank $82:
org !bank_82_freespace_start
;load_pause_map_hook:
;    jsr $943D           ; run hi-jacked instruction (load pause menu map tilemap without items, into $7E4000)
;    jsl update_tilemap  ; (add items to the tilemap)
;    rts

transfer_pause_tilemap_hook:
    ; Copy tilemap from [$00] (original ROM map tilemap) to $703000
    ldx #$0800  ; loop counter
    ldy #$0000  ; offset

    lda #$3000
    sta $03
    lda #$0070
    sta $05

.loop:
    lda [$00], y
    sta [$03], y
    iny
    iny
    dex
    bne .loop

    ; Set source tilemap to $703000 for further processing
    lda #$3000
    sta $00
    lda #$0070
    sta $02

    jsl update_tilemap  ; update tilemap with collected item dots

    lda #$4000  ; run hi-jacked instruction
    rts

warnpc !bank_82_freespace_end

; Free space in any bank:
org $83B300

update_tilemap:
    php

    rep #$30

    lda $1F5B
    asl
    tax
    lda !item_list_sizes, x
    beq .done  ; If there are no items in this area, then we're done.
    sta $06    ; $06 <- loop counter
    lda !item_list_ptrs, x
    tax  ; X <- item data offset

.loop:
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
    bne .loop

.done:
    plp
    rtl