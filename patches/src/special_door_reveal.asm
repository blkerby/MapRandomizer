; See the function `setup_special_door_reveal` in map_tiles.rs.

!bank_85_free_space_start = $85A180
!bank_85_table_end = $85A1D6
!bank_85_free_space_end = $85A280

org $8293E5
    jsl hook_load_pause_tilemap
    nop : nop : nop : nop : nop : nop : nop : nop

org !bank_85_free_space_start
special_door_reveal_table:
; The location of this table must match the address in `setup_special_door_reveal` in map_tiles.rs,
; which is responsible for populating it.
;
; Table format: a sequence of records, each with following data:
;   2 bytes: area (0-5) (or $FFFF to mark end of table)
;   2 bytes: triggering partial revealed address
;      address in bank $70 of partial revealed bit of tile that triggers reveal of neighboring doors
;   2 bytes: triggering partial revealed bitmask:
;      bitmask to be applied to the value at that address (decribed above), to determine if the tile is partially revealed
;   2 bytes: target partial revealed address:
;      address in bank $70 of partial revealed bit of neighboring tile that is to be triggered to reveal a door
;   2 bytes: target partial revealed bitmask:
;      bitmask to be applied to the value at that address (described above), to determine if the tile is partially revealed
;   2 bytes: target tilemap address: address (in bank $7E) where to write replacement tilemap value, when applicable.
;   2 bytes: target tilemap replacement value: tilemap word for showing the partially revealed door

org !bank_85_table_end

hook_load_pause_tilemap:
    php
    phb

    rep #$30
    phk
    plb       ; DB <- $85
    lda #$0000
    sta $06   ; $06 <- offset into special_door_reveal_table
.loop:
    ldx $06
    lda special_door_reveal_table,x
    cmp #$FFFF
    beq .done                        ; exit if end-of-table marker is reached
    cmp $1F5B
    bne .skip                        ; skip record if area does not match

    lda special_door_reveal_table+2,x
    tay                              ; Y <- triggering tile partial revealed address (referencing bank $70)
    lda special_door_reveal_table+4,x
    sta $08                          ; [$08] <- triggering tile partial revealed bitmask
    tyx
    lda $700000,x
    bit $08
    beq .skip                        ; if the triggering tile is not partially revealed, then there's nothing to do.

    ldx $06
    lda special_door_reveal_table+6,x
    tay                              ; Y <- target tile partial revealed address (referencing bank $70)
    lda special_door_reveal_table+8,x
    sta $08                          ; [$08] <- target tile partial revealed bitmask
    tyx
    lda $700000,x
    bit $08
    bne .skip                        ; if the target tile is already partially revealed, then there's nothing to do.

    ; load the replacement tilemap word into the target address, to draw the partially revealed door:
    ldx $06
    lda special_door_reveal_table+10,x
    tay
    lda special_door_reveal_table+12,x
    tyx
    sta $7E0000,x

.skip:
    lda $06
    clc
    adc #$000E                       ; move forward to next record
    sta $06
    bra .loop

.done:
    plb
    plp

    ; run hi-jacked instruction:
    jsl $8091A9                                ; set up DMA transfer
    db $01, $01, $18, $00, $40, $7E, $00, $10  ; dx 01,01,18,7E4000,1000

    rtl

warnpc !bank_85_free_space_end
