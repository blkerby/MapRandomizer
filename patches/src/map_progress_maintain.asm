arch snes.cpu
lorom

!bank_90_freespace_start = $90F700
!bank_90_freespace_end = $90F780

; Expand SRAM from 8 KB to 16 KB.
org $80FFD8
    db 4

; Modify SRAM size check to work for the new SRAM size
org $808695 : LDX #$3FFE
org $8086A7 : LDX #$3FFE
org $8086B5 : LDX #$3FFE
org $8086B8 : STA $704000, x
org $8086C4 : LDX #$3FFE
org $8086D2 : LDX #$3FFE

org $90A98B
    jsr mark_progress

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


;;;; $943D: Load pause menu map tilemap ;;;
org $82943D
load_pause_map_tilemap:
    PHP
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
    LDY #$0000             ; Y = 0 (tilemap index)
    LDX #$0000             ; X = 0 (map data byte index)
    STZ $12                ; $12 = 0 (map data bit subindex)
    CLC

.LOOP_WITHOUT_MAP_DATA:
    ROL $07F7,x               ;\
    BCS .BRANCH_EXPLORED_MAP_TILE ;} If [$07F7 + [X]] & 80h >> [$12] != 0: go to BRANCH_EXPLORED_MAP_TILE
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
    CPX #$0100             ;\
    BMI .LOOP_WITHOUT_MAP_DATA     ;} If [X] < 100h: go to LOOP
    PLP
    RTS                    ; Return

.BRANCH_EXPLORED_MAP_TILE:
    INC $07F7,x
    REP #$30
    LDA [$00],y            ;\
    AND #$FBFF             ;} [$03] + [Y] = [[$00] + [Y]] & ~400h
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
    AND #$FBFF             ; A &= ~400h
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

warnpc !bank_90_freespace_end