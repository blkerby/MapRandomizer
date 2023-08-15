; Certain rooms contain enemies that reinitialize the game RNG: Hoppers, Dessgeegas, Beetoms, lavaquake rocks.
; However, if the room has a music track change, after initialization by the enemies the RNG continues 
; updating while the new track loads (though it is not used during that time). This would throw off the 
; enemy behavior if the randomizer has a track change when vanilla does not, or vice versa. To solve this, 
; we reinitialize the game RNG after the music has loaded, to match the RNG values of the vanilla game.
;
; The RNG value after music loading depends on the enemy type in the room, whether or not there is a music 
; track change (in vanilla), and whether or not there is lava/acid in the room. Given the amount of variables
; that go into this, we just use a table of RNG values by room and/or door (obtained by using a debugger
; and memory watch to see the vanilla RNG value at $05E5 when the music finishes loading, setting a breakpoint 
; at $82E112).

!bank_83_free_space_start = $83B400
!bank_83_free_space_end = $83B500

org $82E66A
    jsl load_music_hook
    LDA #$E6A2
    STA $099C

org !bank_83_free_space_start
load_music_hook:
    ; run hi-jacked code: load new music track if changed
    jsl $82E0D5

    ; set data bank to $83
    phk
    plb 

    ; look for matching entry in table, and if found then use the RNG value from the table.
    ldx #$0000
.loop:
    lda rng_table,x
    beq .done   ; stop if end-of-table marker found
    cmp $079B   ; room pointer
    bne .next
    lda rng_table+2,x
    beq .match  ; catch-all door match
    ldy $078D
    cmp $0004,y
    beq .match  ; door cap X&Y match
.next:
    rep 6 : inx
    bra .loop
.match:
    lda rng_table+4,x
    sta $05E5   ; replace RNG value
.done:
    rtl

; Table of RNG values; each record is 6 bytes:
;   - room pointer (2 bytes)
;   - door cap X & Y (2 bytes), or zero if any door.
;   - RNG value (2 bytes).
; The first matching record will be used, so specific door records must come before catch-all (zero) door values.
rng_table:
    ; Sidehoppers/Dessgeegas:
    dw $9E9F, $2601, $661C    ; Morph Ball Room left
    dw $9E9F, $0000, $3320    ; Morph Ball Room top and right
    dw $9B9D, $0000, $3320    ; Brinstar Pre-Map Room
    dw $9BC8, $0000, $3320    ; Early Supers Room
    dw $9E52, $0000, $3320    ; Green Hill Zone
    dw $9D19, $0000, $3320    ; Big Pink
    dw $9E11, $0000, $3320    ; Pink Brinstar Power Bomb Room (Mission Impossible)
    dw $A130, $0000, $3320    ; Pink Brinstar Hopper Room (Wave Gate)
    dw $A37C, $0000, $3320    ; Beta Power Bomb Room
    dw $A7B3, $0000, $1043    ; Cathedral Entrance
    dw $A815, $0000, $3320    ; Ice Beam Gate Room
    dw $B5D5, $0000, $1043    ; Wasteland
    dw $B40A, $0000, $1043    ; Mickey Mouse Room
    dw $DC19, $0601, $3320    ; (Tourian) Blue Hopper Room left
    dw $DC19, $0316, $661C    ; (Tourian) Blue Hopper Room top
    ; Beetoms:
    dw $99BD, $0000, $08A5    ; Green Pirates Shaft
    dw $9FE5, $0000, $08A5    ; Green Brinstar Beetom Room
    dw $A011, $0000, $08A5    ; Etecoon Energy Tank Room
    dw $A4B1, $0000, $08A5    ; Warehouse Energy Tank Room
    dw $A253, $4601, $AB74    ; Red Tower top left door (from Noob Bridge)
    dw $A253, $0000, $08A5    ; Red Tower other doors
    dw $B106, $0000, $08A5    ; Frog Speedway
    ; Lavaquake rocks:
    dw $AE32, $0000, $FF16    ; Volcano Room
    dw $0000                  ; end-of-table marker

warnpc !bank_83_free_space_end