arch snes.cpu
lorom

!map_station_reveal_type = $90F700  ; 0 = Full reveal,  1 = Partial reveal
!map_reveal_tile_table = $90FA00  ; must match reference in patch.rs
!bank_90_freespace_start = $90FC02
!bank_90_freespace_end = $90FD10


incsrc "constants.asm"

; In the game header, expand SRAM from 8 KB to 16 KB.
org $80FFD8
    db 4

; Modify SRAM size check to work for the new SRAM size.
; We just check that the SRAM can remember the incrementing sequence. 
; No need to check mirroring behavior since this isn't used in the game and doesn't work with some flash carts such
; as Everdrive.
org $808695 : LDX #$3FFE
org $8086A7 : LDX #$3FFE
org $8086B5 : LDX #$3FFE
org $8086B8 : STA $700000, x
org $8086C4 : LDX #$3FFE
org $8086D2 : LDX #$3FFE

org $848CA6
    jsl activate_map_station_hook
    nop : nop

; Continue marking map tiles revealed/explored even if mini-map is disabled:
org $90A923
    nop : nop

; Hook routine that marks the tile above Samus as explored (in Terminator and Croc Speedway)
org $90AB6D
    jsr hook_mark_tile_above

; Hook for normal routine to mark tiles explored (at current Samus location)
; This will also check if mini-map is disabled, and if so, skip the rest of the mini-map drawing routine.
org $90A98B
    jmp mark_progress

org !map_station_reveal_type
    dw $0000  ; default: full reveal

org !bank_90_freespace_start
mark_progress:
    lda $12  ; Samus X map coordinate
    cmp !last_samus_map_x
    bne .moved
    lda $16  ; Samus Y map coordinate
    cmp !last_samus_map_y
    bne .moved

    ; Samus hasn't moved to a new map tile, so skip updating the mini-map.
    ; Only update the flashing to mark Samus' location (unless minimap is disabled)
    rep #$30 : dex  ; run hi-jacked instructions
    lda $05F7
    bne .minimap_disabled  

    lda $05B5         ;\
    and #$0008        ;} If [8-bit frame counter] & 8 = 0:
    bne .flash_off    ;/
    lda $7EC680       ;\
    and #$E3FF        ;} give Samus position in mini-map palette 0 (orange color)
    sta $7EC680       ;/
    bra .done
.flash_off:
    lda $7EC680       ;\
    ora #$0800        ;} give Samus position in mini-map palette 2 (explored map color)
    sta $7EC680       ;/
.done:
    plp
    rtl

.moved:
    ; Samus has moved to a different map tile, so update the mini-map:
    phx
    lda $12
    sta !last_samus_map_x
    lda $16
    sta !last_samus_map_y

    ; convert X from within-area byte index (between $00 and $ff) to an overall byte index (between $00 and $5ff)
    ; (accumulator is 8-bit)
    txa
    xba
    lda $1F5B
    xba
    tax

    ; update map progress in SRAM (same data shared across all save slots):
    lda $702000,x
    ora $AC04,y         ; A |= $80 >> Y
    sta $702000,x

    lda $702700,x
    ora $AC04,y         ; A |= $80 >> Y
    sta $702700,x

    plx
    rep #$30 : dex  ; run hi-jacked instructions

    lda $05F7
    bne .minimap_disabled
    jmp $A98E
.minimap_disabled:
    plp
    rtl

; When map station is activated, fill all map revealed bits for the area:
activate_map_station_hook:
    LDA #$0001 : STA $0789   ; run hi-jacked instructions (set map flag)

    LDX $1F5B
    LDA $D908,X
    STA $700b58,X
    STA $701558,X
    STA $700158,X

    phb
    pea $9090
    plb
    plb

    ; reveal specific tiles in other area maps (e.g. area transition arrows/letters)
    lda $1F5B
    asl
    tax          ; X <- map_area * 2
    lda !map_reveal_tile_table, x
    tay          ; Y <- [map_reveal_tile_table + map_area * 2]

.cross_area_reveal_loop:
    lda $0000,y
    beq .done_cross_area_reveal
    tax            ; X <- address of word containing tile to reveal (relative to base at $702000 and $702700)
    lda $0002,y    ; A <- bitmask of tile to reveal
    ora $702000,x
    sta $702000,x
    lda $0002,y
    ora $702700,x
    sta $702700,x
    iny : iny : iny : iny
    bra .cross_area_reveal_loop

.done_cross_area_reveal:
    plb

    ; now reveal the current area's map:
    lda $1F5B
    xba
    tax          ; X <- map area * $100
    ldy $0080    ; Y <- loop counter (number of words to fill with #$FFFF)

    lda !map_station_reveal_type
    bne .partial_only_loop

.loop:
    lda #$FFFF
    sta $702000, x
    sta $702700, x
    inx
    inx
    dey
    bne .loop
    rtl

.partial_only_loop:
    lda #$FFFF
    sta $702700, x
    ; fully reveal specific tiles that contain area-transition markers,
    ; since those would not show correctly in the partially-revealed palette:
    lda $829727, x
    ora $702000, x
    sta $702000, x
    inx
    inx
    dey
    bne .partial_only_loop
    rtl

hook_mark_tile_above:
    ; run hi-jacked instruction (mark explored tile)
    sta $07F3,x

    ; convert X from within-area byte index (between $00 and $ff) to an overall byte index (between $00 and $5ff)
    ; (accumulator is 8-bit)
    txa
    xba
    lda $1F5B
    xba
    tax
    
    ; mark revealed tile:
    lda $702000-4, x
    ora $AC04,y
    sta $702000-4, x

    rts

warnpc !bank_90_freespace_end

;;;; $943D: Load pause menu map tilemap ;;;
org $82943D
load_pause_map_tilemap:
    PHP

    jsl $8085C6  ; mirror explored bits

    REP #$30
    LDA $1F5B              ; $12 = [area index]
    STA $12              
    ASL A                  ;\
    CLC                    ;|
    ADC $12                ;|
    TAX                    ;|
    LDA $964A,x            ;} $00 = [$964A + [$12] * 3] (source tilemap)
    STA $00                ;|
    LDA $964C,x            ;|
    STA $02                ;/
    BRA .set_dst

warnpc $82945C
org $82945C      ; We keep this instruction in the same place so that item_dots_disappear can hook into it
.set_dst
    LDA #$4000             ;\
    STA $03                ;|
    LDA #$007E             ;} $03 = $7E:4000 (destination tilemap)
    STA $05                ;/
    LDA $12                ;\
    ASL A                  ;|
    TAX                    ;|
    LDX $1F5B              ;\
    SEP #$20

    ; X := X << 8
    txa
    xba
    lda $00
    
    tax
    LDY #$0000             ; Y = 0 (tilemap index)
    STZ $12                ; $12 = 0 (map data bit subindex)
    
    LDA $7ECD52, x         ; load first set of map tile explored bits
    STA $06
    LDA $702000, x         ; load first set of map tile revealed bits (persisted across deaths/reloads)
    STA $26
    LDA $702700, x         ; load first set of map tile partial revealed bits (persisted across deaths/reloads)
    STA $28

    CLC

.LOOP:
;    ROL $07F7,x               ;\
    ROL $06
    BCS .BRANCH_EXPLORED_MAP_TILE ;} If [$07F7 + [X]] & 80h >> [$12] != 0: go to BRANCH_EXPLORED_MAP_TILE
    ROL $26
    BCS .BRANCH_REVEALED_MAP_TILE
    ROL $28
    BCS .BRANCH_PARTIAL_REVEALED_MAP_TILE

    REP #$20
    LDA #$001F             ;\
    STA [$03],y            ;} [$03] + [Y] = 001Fh (blank tile)

.BRANCH_NEXT:
    SEP #$20
    INY                    ;\
    INY                    ;} Y += 2
    INC $12                ; Increment $12
    LDA $12                ;\
    CMP #$08               ;} If [$12] < 8: go to LOOP
    BMI .LOOP ;/
    STZ $12                ; $12 = 0
    INX                    ; Increment X
    lda $7ECD52, x         ; load next set of map tile explored bits
    sta $06
    lda $702000, x         ; load next set of map tile revealed bits (persisted across deaths/reloads)
    sta $26
    lda $702700, x         ; load next set of map tile partial revealed bits (persisted across deaths/reloads)
    sta $28
    txa
    bne .LOOP     ;} If [X] % $100 != 0: go to LOOP
    PLP
    RTS                    ; Return

.BRANCH_REVEALED_MAP_TILE:
    ROL $28
    REP #$30
    LDA [$00],y            ;\
    STA [$03],y            ;/
    BRA .BRANCH_NEXT     ; Go to BRANCH_NEXT

.BRANCH_PARTIAL_REVEALED_MAP_TILE:
    REP #$30
    LDA [$00],y            ;\
    AND #$EFFF             ; Use palette 3 (instead of 6)
    ORA #$0400             ; 
    STA [$03],y            ;/
    BRA .BRANCH_NEXT     ; Go to BRANCH_NEXT

.BRANCH_EXPLORED_MAP_TILE:
    ROL $26
    ROL $28
    REP #$30
    LDA [$00],y            ;\ Use palette 2 (instead of 6)
    AND #$EFFF             ;} [$03] + [Y] = [[$00] + [Y]] & ~1000h
    STA [$03],y            ;/
    BRA .BRANCH_NEXT     ; Go to BRANCH_NEXT


warnpc $829628

; For HUD mini-map drawing, load map revealed bits (persisted across deaths/reloads) 
; in place of map data bits (which are irrelevant since whole map is revealed):
org $90A9C1
    NOP
    XBA                    ; A := A << 8
    CLC
    ADC #$2000
    STA $09                ; $09 = $702000 + [area index] * $100
    STA $0F                ; $0F = $702000 + [area index] * $100
    LDA #$0070             
    STA $0B                
    ; PC should now be exactly $90A9D0:
    print "$90A9D0 =? ", pc

; use palette 6 for unexplored tile in HUD minimap
org $90AAB4 : ORA #$3800  ; row 0, was: ORA #$2C00
org $90AADB : ORA #$3800  ; row 1, was: ORA #$2C00
org $90AB18 : ORA #$3800  ; row 2, was: ORA #$2C00

; Patch HUD mini-map drawing to use map revealed bits instead of map data bits
; Vanilla logic: tile is non-blank if map station obtained AND map data bit is set
; New logic: tile is non-blank if map revealed bit is set
; (There's some redundant code here; it could be optimized.)
org $90AAAC : BRA $03   ; Row 0
org $90AAD3 : BRA $03   ; Row 1
org $90AB10 : BRA $03   ; Row 2

; Patch "Determine map scroll limits" to be based on map partial revealed bits instead of map explored bits:
org $829EC6
    lda $1F5B
    clc
    xba
    adc #$2700
    sta $06
    lda #$0070
    sta $08         ; $06 <- $702000 + area index * $100
    jmp $9EEA
    

;$82:9EC6 AD 89 07    LDA $0789  [$7E:0789]  ;\
;$82:9EC9 F0 15       BEQ $15    [$9EE0]     ;} If area map has been collected:
;$82:9ECB A9 82 00    LDA #$0082             ;\
;$82:9ECE 85 08       STA $08    [$7E:0008]  ;} $08 = $82
;$82:9ED0 A9 17 97    LDA #$9717             ;\
;$82:9ED3 85 06       STA $06    [$7E:0006]  ;|
;$82:9ED5 AD 9F 07    LDA $079F  [$7E:079F]  ;|
;$82:9ED8 0A          ASL A                  ;} $06 = [$82:9717 + [area index] * 2] (map data pointer)
;$82:9ED9 A8          TAY                    ;|
;$82:9EDA B7 06       LDA [$06],y[$82:9717]  ;|
;$82:9EDC 85 06       STA $06    [$7E:0006]  ;/
;$82:9EDE 80 0A       BRA $0A    [$9EEA]
;
;$82:9EE0 A9 00 00    LDA #$0000             ;\ Else (area map has not been collected):
;$82:9EE3 85 08       STA $08    [$7E:0008]  ;|
;$82:9EE5 A9 F7 07    LDA #$07F7             ;} $06 = $00:07F7 (map tiles explored)
;$82:9EE8 85 06       STA $06    [$7E:0006]  ;/

; Patches below re-enable mini-map with boss deaths
; ridley
org $8081b1
    jsl fix_boss

; croc
org $a490c4
    jsr fix_minimap

; dray
org $a592d7
    jsl fix_boss

; kraid
org $a7c825
    jsl fix_boss

; phant
org $a7db7e
    jsl fix_boss

; mb
org $a9b275
    jsl fix_mb : nop : nop

!bank_a4_free_space_start = $a4f6d0
!bank_a4_free_space_end = $a4f6f0

org !bank_a4_free_space_start
fix_minimap:
    stz $5f7                      ; enable mini-map
    lda #$ffff
    sta !last_samus_map_x         ; null x,y to induce update
    sta !last_samus_map_y
    rts

fix_boss:
    jsr fix_minimap
    lda $7ed828,x                 ; replaced code
    rtl

fix_mb:
    jsr fix_minimap
    ldy #$9534                    ; replaced code
    ldx #$0122
    rtl

warnpc !bank_a4_free_space_end
