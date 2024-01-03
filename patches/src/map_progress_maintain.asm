arch snes.cpu
lorom

!bank_90_freespace_start = $90F700
!bank_90_freespace_end = $90F780

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

; Hook for normal routine to mark tiles explored (at current Samus location)
org $90A98B
    jsr mark_progress

; Hook for special routine to mark tiles (used when entering boss rooms)
org $90A8E8
    jsr mark_tile_explored_hook

org !bank_90_freespace_start
mark_progress:
    phx

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

    plx
    rep #$30 : dex  ; run hi-jacked instructions
    rts

; When map station is activated, fill all map revealed bits for the area:
activate_map_station_hook:
    LDA #$0001 : STA $0789   ; run hi-jacked instructions (set map flag)
    
    lda $1F5B
    xba
    tax          ; X <- map area * $100
    ldy $0080    ; Y <- loop counter (number of words to fill with #$FFFF)
.loop    
    lda #$FFFF
    sta $702000, x
    inx
    inx
    dey
    bne .loop
    
    rtl


mark_tile_explored_hook:
    STA $07F7,x   ; run hi-jacked instruction (mark tile explored)

    ; Also mark tile revealed (persists after deaths/reloads)
    lda $1F5B  ; load current area
    xba
    txa  ; only low 8-bits of X transferred to low 8 bits of A
    tax  ; full 16-bits of A transferred to X:  X <- area * $100 + offset
    lda $702000,x
    ora $AC04,y
    sta $702000,x

    ; Also mark tile partial revealed (persists after deaths/reloads)
    lda $702700,x
    ora $AC04,y
    sta $702700,x

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
    LDA $7ED908,x          ;|
    AND #$00FF             ;} If area map collected: go to BRANCH_MAP_COLLECTED
    BNE .BRANCH_MAP_COLLECTED
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

.LOOP_WITHOUT_MAP_DATA:
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

.BRANCH_NEXT_WITHOUT_MAP_DATA:
    SEP #$20
    INY                    ;\
    INY                    ;} Y += 2
    INC $12                ; Increment $12
    LDA $12                ;\
    CMP #$08               ;} If [$12] < 8: go to LOOP
    BMI .LOOP_WITHOUT_MAP_DATA ;/
    STZ $12                ; $12 = 0
    INX                    ; Increment X
    lda $7ECD52, x         ; load next set of map tile explored bits
    sta $06
    lda $702000, x         ; load next set of map tile revealed bits (persisted across deaths/reloads)
    sta $26
    lda $702700, x         ; load next set of map tile partial revealed bits (persisted across deaths/reloads)
    sta $28
    txa
    bne .LOOP_WITHOUT_MAP_DATA     ;} If [X] % $100 != 0: go to LOOP
    PLP
    RTS                    ; Return

.BRANCH_REVEALED_MAP_TILE:
    ROL $28
    REP #$30
    LDA [$00],y            ;\
    STA [$03],y            ;/
    BRA .BRANCH_NEXT_WITHOUT_MAP_DATA     ; Go to BRANCH_NEXT_WITHOUT_MAP_DATA

.BRANCH_PARTIAL_REVEALED_MAP_TILE:
    REP #$30
    LDA [$00],y            ;\
    AND #$EFFF             ; Use palette 3 (instead of 6)
    ORA #$0400             ; 
    STA [$03],y            ;/
    BRA .BRANCH_NEXT_WITHOUT_MAP_DATA     ; Go to BRANCH_NEXT_WITHOUT_MAP_DATA

.BRANCH_EXPLORED_MAP_TILE:
    ROL $26
    ROL $28
    REP #$30
    LDA [$00],y            ;\ Use palette 2 (instead of 6)
    AND #$EFFF             ;} [$03] + [Y] = [[$00] + [Y]] & ~1000h
    STA [$03],y            ;/
    BRA .BRANCH_NEXT_WITHOUT_MAP_DATA     ; Go to BRANCH_NEXT_WITHOUT_MAP_DATA

.BRANCH_MAP_COLLECTED:
    REP #$30
    LDA #$0000             ;\
    STA $0B                ;|
    LDA #$07F7             ;} $09 = $00:07F7 (map tiles explored)
    STA $09                ;/
    LDA [$09]              ;\
    XBA                    ;} $28 = [[$09]] << 8 | [[$09] + 1]
    STA $28                ;/
    INC $09                ;\
    INC $09                ;} $09 += 2
    LDY #$0000             ; Y = 0 (tilemap index)
    LDX #$0010             ; X = 10h

.LOOP_WITH_MAP_DATA
    LDA [$00],y            ; A = [[$00] + [Y]]
    ASL $28                ;\
    BCC .not_explored       ;} If [$28] & (1 << [X]-1) != 0:
    AND #$EFFF             ; A &= ~1000h
.not_explored:
    STA [$03],y            ; [$03] + [Y] = [A]
    DEX                    ; Decrement X
    BNE .next                ; If [X] = 0:
    LDX #$0010             ; X = 10h
    LDA [$09]              ;\
    XBA                    ;} $28 = [[$09]] << 8 | [[$09] + 1]
    STA $28                ;/
    INC $09                ;\
    INC $09                ;} $09 += 2
.next:
    INY                    ;\
    INY                    ;} Y += 2
    CPY #$1000             ;\
    BMI .LOOP_WITH_MAP_DATA    ;} If [Y] < 1000h: go to LOOP_WITH_MAP_DATA
    PLP
    RTS

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
; (We make minimal changes to the code here, leaving redundant computations, to minimize changes to timings)
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